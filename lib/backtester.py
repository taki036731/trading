import datetime as dt
import logging
from dataclasses import dataclass, fields, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import vectorbt as vbt

import lib.data_saver as ds
from lib.metrics import (
    DEFAULT_MIN_FILL_RATE,
    entry_order_stats,
    portfolio_stats,
    warn_low_fill_rate,
)
from lib.strategies.exits import AbstractStopLoss, AbstractTakeProfit
from lib.strategies.indicators import AbstractIndicator
from lib.strategies.position_sizing import AbstractPositionSizer
from lib.strategies.signals import AbstractSignal

logger = logging.getLogger(__name__)

# 日本株の売買単位（100株単元）
UNIT_SHARES = 100

# 売買代金に対する手数料の割合（片道）。国内株は手数料無料のネット証券が多いため0とする
DEFAULT_FEES = 0.0
# 約定価格に対するスリッページの割合（片道）
DEFAULT_SLIPPAGE = 0.001

# 日本株向けの約定ルール。単一銘柄・複数銘柄のどちらのバックテストでも同じ値を使う。
# 100株単元の制約（size_granularity / min_size）は、日ごとに値が変わるため PreparedSignals で渡す
# 手数料・スリッページは差し替えられるように Backtester の引数で渡す
JP_MARKET_PARAMS: dict[str, Any] = {
    "direction": "longonly",
    # SL/TP ラインの基準を約定価格(始値)にする。
    # 既定値はエントリー日の終値で、想定したラインからずれる
    "stop_entry_price": "fillprice",
    # SL/TP の決済にもスリッページをかける。
    # 既定値の stoplimit ではスリッページが0になり、成績が楽観的に出る。
    # SL と TP を分けて指定できないため、実際には指値で出せる TP にもかかる（保守的な側に倒す）
    "stop_exit_price": "stopmarket",
    "freq": "D",  # 日足データ
    # 注文の記録を残す。1単元に届かないなどで捨てられた注文は取引履歴に残らないため、
    # portfolio.logs から数える（lib.metrics.entry_order_stats）。売買の結果は変わらない
    "log": True,
}

# エントリー当日を2行に分けるときの、2行目（始値より後の値動き）の時刻
SPLIT_BAR_OFFSET = pd.Timedelta(hours=12)

# 評価期間の開始日・終了日に渡せる型（pd.Timestamp に変換できるもの）
DateLike = str | dt.date | pd.Timestamp


def period_mask(
    index: pd.Index, start: DateLike | None, end: DateLike | None
) -> np.ndarray:
    """
    インデックスのうち、評価期間 [start, end]（両端を含む）に入る行を返します。

    Args:
        index (pd.Index): 日付のインデックス。
        start (DateLike | None): 評価期間の開始日。None の場合は先頭から。
        end (DateLike | None): 評価期間の終了日。None の場合は末尾まで。

    Returns:
        np.ndarray: 評価期間に入る行が True の bool 配列。

    Raises:
        ValueError: start が end より後の場合、または評価期間に行がない場合。
    """
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"エラー: 評価期間の開始日 {start_ts} が終了日 {end_ts} より後です。"
        )
    mask = np.ones(len(index), dtype=bool)
    if start_ts is not None:
        mask &= np.asarray(index >= start_ts)
    if end_ts is not None:
        mask &= np.asarray(index <= end_ts)
    if not mask.any():
        raise ValueError(
            f"エラー: 評価期間 [{start_ts}, {end_ts}] にデータがありません。"
        )
    return mask


@dataclass
class PreparedSignals:
    """
    1銘柄分の、vectorbt の from_signals に渡す入力一式。

    複数銘柄（資金共有モード）では、MultiBacktester が銘柄ごとの PreparedSignals を
    列=銘柄の DataFrame にまとめ、from_signals のキーワード引数として直接渡します。
    """

    close: pd.Series
    # SL/TP をザラ場の値動きで判定するために OHLC を渡す。
    # 渡さないと終値だけで判定され、約定価格も終値になってしまう
    open: pd.Series
    high: pd.Series
    low: pd.Series
    price: pd.Series  # 約定価格（翌日始値）
    entries: pd.Series
    exits: pd.Series
    # シフト済みの損切り・利確幅（比率）。NaN はストップなし。
    # vectorbt では 0 でもストップは有効で、ラインが約定価格（建値）になるため、「なし」に 0 は使わない
    sl_stop: pd.Series
    tp_stop: pd.Series
    # 損切りをトレーリングストップにするか。True なら SL ラインが保有中の最高値（High）を基準に上がっていく。
    # 損切り幅はエントリー日の sl_stop で固定される（vectorbt の仕様）
    sl_trail: bool
    size: pd.Series
    size_type: int
    # 100株単元を調整後の株数で表したもの。エントリーシグナルのある日だけ値を持つ
    size_granularity: pd.Series
    min_size: pd.Series

    def to_vbt_kwargs(self) -> dict:
        """
        from_signals のキーワード引数に変換します。

        Returns:
            dict: close / open / high / low / price / entries / exits / sl_stop / tp_stop / sl_trail / size /
                size_type / size_granularity / min_size。
        """
        return {
            "close": self.close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "price": self.price,
            "entries": self.entries,
            "exits": self.exits,
            "sl_stop": self.sl_stop,
            "tp_stop": self.tp_stop,
            "sl_trail": self.sl_trail,
            "size": self.size,
            "size_type": self.size_type,
            "size_granularity": self.size_granularity,
            "min_size": self.min_size,
        }

    def between(
        self, start: DateLike | None, end: DateLike | None
    ) -> "PreparedSignals":
        """
        評価期間 [start, end]（両端を含む）の行だけを切り出したものを返します。

        シフト済みの値を行ごと切り出すだけなので、評価期間の初日のシグナルや SL/TP 幅には、
        前日（評価期間より前のウォームアップ期間）の終値までの情報が入ります（未来の情報は入りません）。
        足の分割（split_entry_day_bars）より前に行います。

        Args:
            start (DateLike | None): 評価期間の開始日。None の場合は先頭から。
            end (DateLike | None): 評価期間の終了日。None の場合は末尾まで。

        Returns:
            PreparedSignals: 評価期間の行だけの入力。start と end がともに None なら self をそのまま返します。

        Raises:
            ValueError: start が end より後の場合、または評価期間に行がない場合。
        """
        if start is None and end is None:
            return self
        mask = period_mask(self.close.index, start, end)
        sliced = {
            f.name: value[mask]
            for f in fields(self)
            if isinstance(value := getattr(self, f.name), pd.Series)
        }
        return replace(self, **sliced)

    def split_entry_day_bars(self, slippage: float) -> "PreparedSignals":
        """
        エントリー当日に SL/TP ラインへ届く日の足を、2行に分けたものを返します。

        vectorbt はエントリーした足では SL/TP を判定しないため、始値で買った当日に
        ラインへ届いても手仕舞われません。該当日を次の2行に分け、2行目で判定させます。

        - 1行目（元の日時）: 始値だけの足（O=H=L=C=始値）。ここで約定する
        - 2行目（元の日時 + SPLIT_BAR_OFFSET）: 始値より後の値動き（始値と実際の H/L/C）。シグナルは出さない

        シフト済みのシグナル・SL/TP 幅・サイズは、元の日時の行にそのまま残します（行をずらさない）。
        そのため、分割は必ず prepare() の後に行います。
        同じ日に SL と TP の両方に届いた場合は、vectorbt の判定順で SL が優先されます。
        SL/TP 幅が NaN の日は、その側のストップがないものとして扱います。

        トレーリングストップ（sl_trail=True）では、ラインへ届いたかに関係なく、エントリーのある日を全て分けます。
        vectorbt はエントリーした足でトレーリングの最高値を更新しないため、分けないとエントリー当日の高値が
        最高値に入らず、分けた日とラインの上がり方がそろわないためです。2行目では、判定の後に最高値を更新するので、
        当日の高値で上がったラインが効くのは翌日からです（同じ足の中で未来の値は使いません）。

        Args:
            slippage (float): 約定価格に対するスリッページの割合。SL/TP ラインは
                スリッページ込みの約定価格が基準になるため、判定にも使います。

        Returns:
            PreparedSignals: 該当日を分割した入力。該当日がなければ self をそのまま返します。
        """
        # ラインの基準は約定価格（stop_entry_price="fillprice"）。始値基準だと、ラインの近くで分割が漏れる
        fill = self.open * (1 + slippage)
        # NaN はストップなし。0 は vectorbt では建値のストップとして有効なので、「あり」として扱う
        sl_on = self.sl_stop.notna()
        touched = self.entries & (
            # トレーリングでは、エントリー当日の高値を最高値に入れるため、SL のある日は全て分ける
            (sl_on & (self.sl_trail | (self.low <= fill * (1 - self.sl_stop))))
            | (self.tp_stop.notna() & (self.high >= fill * (1 + self.tp_stop)))
        )
        if not touched.any():
            return self

        days = cast(pd.DatetimeIndex, self.open.index)[touched.to_numpy()]
        # 元のインデックスと日時の単位（ms など）をそろえる
        rest_index = (days + SPLIT_BAR_OFFSET).as_unit(days.unit)

        def _split(first: pd.Series, rest: pd.Series) -> pd.Series:
            """元の行（first）に、該当日の2行目（rest の値）を差し込みます。"""
            return pd.concat([first, rest[touched].set_axis(rest_index)]).sort_index()

        index = self.open.index
        no_signal = pd.Series(False, index=index)
        flat = touched.to_numpy()  # 1行目を始値だけの足にする日

        return PreparedSignals(
            close=_split(self.close.mask(flat, self.open), self.close),
            open=_split(self.open, self.open),
            high=_split(self.high.mask(flat, self.open), self.high),
            low=_split(self.low.mask(flat, self.open), self.low),
            price=_split(self.price, self.open),
            entries=_split(self.entries, no_signal),
            exits=_split(self.exits, no_signal),
            sl_stop=_split(self.sl_stop, self.sl_stop),
            tp_stop=_split(self.tp_stop, self.tp_stop),
            sl_trail=self.sl_trail,
            size=_split(self.size, pd.Series(np.nan, index=index)),
            size_type=self.size_type,
            size_granularity=_split(self.size_granularity, pd.Series(np.nan, index=index)),
            min_size=_split(self.min_size, pd.Series(0.0, index=index)),
        )


def _check_duplicate_output_columns(indicators: list[AbstractIndicator]) -> None:
    """
    複数のインジケーターが同じ名前のカラムを出力しないかを検証します。
    データを読む前（Backtester の初期化時）に気付けるようにするためのものです。

    Args:
        indicators (list[AbstractIndicator]): 検証するインジケーター。

    Raises:
        ValueError: 同じ名前のカラムを出力するインジケーターが複数ある場合。
    """
    owners: dict[str, AbstractIndicator] = {}
    for indicator in indicators:
        for col in indicator.output_columns:
            if col in owners:
                raise ValueError(
                    f"エラー: {type(owners[col]).__name__} と {type(indicator).__name__} が、"
                    f"同じカラム '{col}' を出力します。出力するカラム名を変えてください。"
                )
            owners[col] = indicator


class Backtester:
    """
    売買戦略のバックテストを実行・管理するクラス。

    インジケーターの計算、売買シグナルの生成、ポジションサイジング、
    および vectorbt を使用したシミュレーションの実行を担います。
    """

    def __init__(
        self,
        *,
        indicators: list[AbstractIndicator],
        entries: list[AbstractSignal],
        exits: list[AbstractSignal],
        tp: AbstractTakeProfit | None = None,
        sl: AbstractStopLoss | None = None,
        position_sizer: AbstractPositionSizer,
        init_cash: float = 1000000.0,
        fees: float = DEFAULT_FEES,
        slippage: float = DEFAULT_SLIPPAGE,
        min_fill_rate: float | None = DEFAULT_MIN_FILL_RATE,
    ):
        """
        Backtester を初期化します。

        Args:
            indicators (list[AbstractIndicator]): 計算するインジケーターのリスト。
            entries (list[AbstractSignal]): エントリー（買い）条件シグナルのリスト。
            exits (list[AbstractSignal]): エグジット（売り）条件シグナルのリスト。
            tp (AbstractTakeProfit | None): 利確ロジックの実装。None の場合は利確しません。
            sl (AbstractStopLoss | None): 損切りロジックの実装。None の場合は損切りしません。
            position_sizer (AbstractPositionSizer): ポジションサイズ決定ロジック。
            init_cash (float): 初期資金。デフォルトは 1,000,000.0。
            fees (float): 売買代金に対する手数料の割合（片道）。デフォルトは DEFAULT_FEES。
            slippage (float): 約定価格に対するスリッページの割合（片道）。デフォルトは DEFAULT_SLIPPAGE。
            min_fill_rate (float | None): エントリー注文の約定率がこれを下回ったら、run() で警告を出します。
                デフォルトは DEFAULT_MIN_FILL_RATE。None の場合は警告しません。

        Raises:
            ValueError: 損切り幅を必要とするサイザー（requires_sl_pct=True）に、sl=None を組み合わせた場合。
                または、複数のインジケーターが同じ名前のカラムを出力する場合。
        """
        if sl is None and position_sizer.requires_sl_pct:
            raise ValueError(
                f"{type(position_sizer).__name__} は損切り幅を使うため、sl を省略できません。"
            )
        _check_duplicate_output_columns(indicators)
        self.indicators = indicators
        self.entries = entries
        self.exits = exits
        self.tp = tp
        self.sl = sl
        self.position_sizer = position_sizer
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.min_fill_rate = min_fill_rate
        self.portfolio: vbt.Portfolio | None = None
        self.df: pd.DataFrame | None = None
        self.entries_series: pd.Series | None = None
        self.exits_series: pd.Series | None = None

    @property
    def cost_params(self) -> dict[str, Any]:
        """
        取引コストを、from_signals のキーワード引数で返します。

        Returns:
            dict[str, Any]: fees / slippage。
        """
        return {"fees": self.fees, "slippage": self.slippage}

    def _generate_signals(
        self, df: pd.DataFrame, signals: list[AbstractSignal]
    ) -> pd.Series:
        """
        複数のシグナル条件を統合し、すべてを満たす箇所を判定します。

        Args:
            df (pd.DataFrame): 株価および指標データ。
            signals (list[AbstractSignal]): 判定するシグナルオブジェクトのリスト。

        Returns:
            pd.Series: すべての条件を満たす場合に True となるブール値Series。
        """
        if not signals:
            return pd.Series(False, index=df.index)
        results = [s.generate(df) for s in signals]
        return pd.concat(results, axis=1).all(axis=1)

    def prepare(
        self,
        df: pd.DataFrame,
        name: str = "",
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> PreparedSignals:
        """
        シミュレーションの直前までの処理を行い、vectorbt に渡す入力をまとめて返します。
        入力の df は書き換えず、コピーにインジケーターの列を追加して使います。

        1. インジケーターの計算
        2. 動的な利確・損切り幅の算出（翌日約定用に1日シフト）
        3. エントリー・エグジットのシグナル生成
        4. ポジションサイズと100株単元の決定
        5. 評価期間 [start, end] の行の切り出し

        1〜4 は df の全期間で計算し、その後で評価期間を切り出します。評価期間より前の行は、
        インジケーターのウォームアップにだけ使われ、売買にも年率の指標の計算にも入りません。
        評価期間の初日にインジケーターの列や損切り幅が NaN の場合は、ウォームアップが足りないと警告します。

        複数銘柄を一括で検証する MultiBacktester からも、この処理を共通で使います。
        サイズがサイザーの上限（max_size）に達したエントリーがあれば、件数を警告します。

        Args:
            df (pd.DataFrame): バックテストに使用する株価データ（書き換えません）。
            name (str): 警告に付ける名前（銘柄・パラメータなど）。
            start (DateLike | None): 評価期間の開始日（その日を含む）。None の場合は df の先頭から。
            end (DateLike | None): 評価期間の終了日（その日を含む）。None の場合は df の末尾まで。

        Returns:
            PreparedSignals: 評価期間の、from_signals に渡す価格・シグナル・SL/TP・サイズ。

        Raises:
            ValueError: start が end より後の場合、または評価期間にデータがない場合。
        """
        return self._prepare(df.copy(), name, start, end)

    def _prepare(
        self,
        df: pd.DataFrame,
        name: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> PreparedSignals:
        """
        prepare() の本体です。df を直接書き換えて、インジケーターの列を追加します。

        Args:
            df (pd.DataFrame): 株価データ。呼び出し元でコピーしたものを渡します。
            name (str): 警告に付ける名前（銘柄・パラメータなど）。
            start (DateLike | None): 評価期間の開始日。None の場合は df の先頭から。
            end (DateLike | None): 評価期間の終了日。None の場合は df の末尾まで。

        Returns:
            PreparedSignals: 評価期間の、from_signals に渡す価格・シグナル・SL/TP・サイズ。
        """
        # ---------------------------------------------------------
        # 1. インジケーターの計算 (ベクトル演算で一括処理)
        # ---------------------------------------------------------
        for indicator in self.indicators:
            indicator.generate(df)

        # ---------------------------------------------------------
        # 2. 動的な利確・損切り幅の設定
        # ---------------------------------------------------------
        # 前日の終値を基に、翌日の損切り・利確%を決める。
        # そのため、StopLoss / TakeProfit のパーセンテージを「翌日約定用」にシフトする。
        # 先頭の行や指標の計算前の期間は NaN（ストップなし）のまま残す。
        # 0 で埋めると、vectorbt では建値のストップになる
        no_stop = pd.Series(np.nan, index=df.index)
        sl_pct = (self.sl.generate(df) if self.sl else no_stop).shift(1)
        tp_pct = (self.tp.generate(df) if self.tp else no_stop).shift(1)

        # ---------------------------------------------------------
        # 3. エントリーとエグジットのシグナルを生成
        # ---------------------------------------------------------
        entries = self._generate_signals(df, self.entries)
        exits = self._generate_signals(df, self.exits)

        # ---------------------------------------------------------
        # 4. ポジションサイズの計算
        # ---------------------------------------------------------
        # サイズ計算にもDay TのSL%を渡す(shift(1)した配列を渡す)
        size = self.position_sizer.generate(df, sl_pct=sl_pct)

        # 100株単元の制約はエントリーの日だけにかける。
        # 単元を調整後の株数で表すと配当落ち・分割のたびに値が変わるため、
        # エグジットの日にもかけると、保有株数が単元で割り切れず全量を売れなくなる。
        # （制約のない日の売りは全量になる。保有中にエントリーシグナルが出た日に
        #   損切り・利確が発動し、かつ保有中に単元が変わっていた場合だけは、全量を売れないことがある）
        lot = self._lot_size(df)
        size_granularity = lot.where(entries)
        min_size = lot.where(entries, 0.0)

        prepared = PreparedSignals(
            close=df["Close"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            price=df["Open"],
            entries=entries,
            exits=exits,
            sl_stop=sl_pct,
            tp_stop=tp_pct,
            sl_trail=self.sl.is_trailing if self.sl else False,
            size=size,
            size_type=self.position_sizer.size_type,
            size_granularity=size_granularity,
            min_size=min_size,
        )

        # ---------------------------------------------------------
        # 5. 評価期間の切り出し
        # ---------------------------------------------------------
        # シフトなどを全期間で済ませてから切り出す（切り出してからシフトすると、初日の情報が欠ける）
        prepared = prepared.between(start, end)
        if start is not None:
            self._warn_short_warmup(df, prepared, name)
        # 上限に達した件数は、評価期間のエントリーで数える
        self._warn_capped_size(prepared.size, prepared.entries, name)
        return prepared

    def _warn_short_warmup(
        self, df: pd.DataFrame, prepared: PreparedSignals, name: str
    ) -> None:
        """
        評価期間の初日に、インジケーターの列や損切り幅がまだ計算できていない（NaN）場合に警告します。

        評価期間の前のデータ（ウォームアップ期間）が短いと、評価期間の初めのうちはシグナルやサイズが出ず、
        取引の機会が実際より少なくなります。

        Args:
            df (pd.DataFrame): インジケーターの列を追加した、全期間の株価データ。
            prepared (PreparedSignals): 評価期間を切り出した入力。
            name (str): 警告に付ける名前（銘柄・パラメータなど）。
        """
        first_day = prepared.close.index[0]
        missing = [
            col
            for indicator in self.indicators
            for col in indicator.output_columns
            if pd.isna(df[col].loc[first_day])
        ]
        if self.sl is not None and pd.isna(prepared.sl_stop.iloc[0]):
            missing.append("損切り幅")
        if not missing:
            return
        prefix = f"[{name}] " if name else ""
        logger.warning(
            f"{prefix}評価期間の初日（{first_day}）に、まだ計算できていない値があります: "
            f"{', '.join(missing)}。評価期間の前のデータ（ウォームアップ期間）を長くしてください。"
        )

    def _warn_capped_size(self, size: pd.Series, entries: pd.Series, name: str) -> None:
        """
        エントリーのある日のうち、サイズがサイザーの上限（max_size）に達した件数を警告します。

        上限に達したエントリーは、サイザーの想定（RiskPercentageSizer なら risk_per_trade）より
        小さいリスクしか取っていません。ちょうど上限の日も数えます。

        Args:
            size (pd.Series): 上限を切った後のポジションサイズ。
            entries (pd.Series): エントリーシグナル。
            name (str): 警告に付ける名前（銘柄・パラメータなど）。
        """
        max_size = self.position_sizer.max_size
        if max_size is None:
            return
        n_entries = int(entries.sum())
        n_capped = int((entries & (size >= max_size)).sum())
        if n_capped == 0:
            return
        prefix = f"[{name}] " if name else ""
        logger.warning(
            f"{prefix}ポジションサイズが上限（{max_size}）に達したエントリーが "
            f"{n_capped}/{n_entries} 件あります。これらの実際のリスクは、サイザーの想定より小さくなります。"
        )

    @staticmethod
    def _lot_size(df: pd.DataFrame) -> pd.Series:
        """
        100株単元を、調整後価格で数えた株数に換算します。

        シミュレーションは調整後価格で行うため、分割・配当で価格が下方修正された過去ほど、
        実際の100株は調整後の株数では多くなります（例: 1:5 分割前の100株は、調整後の500株）。
        lot × 調整後の始値 = 100 × 実際の始値 となり、1単元の金額が当時の実際の金額と一致します。
        約定は始値で行うため、始値どうしの比で換算します。

        Args:
            df (pd.DataFrame): 株価データ。'RawOpen'（当時の実際の始値）があれば換算に使います。

        Returns:
            pd.Series: 各日の1単元の株数（調整後）。'RawOpen' がない場合や換算できない日は100。
        """
        if "RawOpen" not in df.columns:
            return pd.Series(float(UNIT_SHARES), index=df.index)
        lot = UNIT_SHARES * df["RawOpen"] / df["Open"]
        return lot.where(lot > 0).fillna(float(UNIT_SHARES))

    @staticmethod
    def _orders_readable(portfolio: vbt.Portfolio) -> pd.DataFrame:
        """
        注文履歴を、読みやすい形式の DataFrame で返します。

        Args:
            portfolio (vbt.Portfolio): バックテスト結果。

        Returns:
            pd.DataFrame: 注文履歴。
        """
        # vectorbt の cached_property を Pylance が関数と見なすため、このルールだけ無視する
        return portfolio.orders.records_readable  # pyright: ignore[reportFunctionMemberAccess]

    def run(
        self,
        df: pd.DataFrame,
        name: str = "",
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> vbt.Portfolio:
        """
        バックテストを実行します。

        prepare() で作った入力を vectorbt に渡してシミュレーションします。
        start / end を指定すると、評価期間 [start, end] だけをシミュレーションします。
        口座は評価期間の初日に初期資金で始まるので、年率の指標やエクスポージャーも評価期間だけで計算されます。
        評価期間より前の df の行は、インジケーターのウォームアップにだけ使います。
        エントリー当日に SL/TP ラインへ届く日は、足を2行に分けてから渡します（split_entry_day_bars）。
        そのため、Portfolio のインデックスには、元の日付に加えて該当日の 12:00 の行が入ります。
        エントリー注文の約定率が min_fill_rate を下回った場合は、警告を出します。

        Args:
            df (pd.DataFrame): バックテストに使用する株価データ（書き換えません）。
            name (str): 警告に付ける名前（銘柄・パラメータなど）。
            start (DateLike | None): 評価期間の開始日（その日を含む）。None の場合は df の先頭から。
            end (DateLike | None): 評価期間の終了日（その日を含む）。None の場合は df の末尾まで。

        Returns:
            vbt.Portfolio: バックテスト結果を含む vectorbt の Portfolio オブジェクト。

        Raises:
            ValueError: start が end より後の場合、または評価期間にデータがない場合。
        """
        # 入力は書き換えず、インジケーターの列を追加したコピーを self.df に残す（チャートの出力で使う）
        df = df.copy()
        prepared = self._prepare(df, name, start, end)

        # 状態を保持 (参照のみ)。チャートを Portfolio とそろえるため、評価期間の行だけを残す
        self.df = df.loc[prepared.close.index]
        self.entries_series = prepared.entries
        self.exits_series = prepared.exits

        # 状態として保持するシグナルは日次のまま。分割はシミュレーションの入力にだけかける
        self.portfolio = vbt.Portfolio.from_signals(
            **prepared.split_entry_day_bars(self.slippage).to_vbt_kwargs(),
            init_cash=self.init_cash,  # 初期資金
            **self.cost_params,
            **JP_MARKET_PARAMS,
        )
        warn_low_fill_rate(
            entry_order_stats(self.portfolio), self.min_fill_rate, label=name
        )
        return self.portfolio

    def display_stats(self):
        """
        バックテストの結果サマリと最新の注文履歴を表示します。
        """
        if self.portfolio is None:
            logger.warning(
                "バックテストが実行されていません。先に run() を呼び出してください。"
            )
            return

        print("-" * 40)
        print("【バックテスト結果サマリ】")
        # 勝率、プロフィットファクター、ドローダウンなどが一括で計算されます。
        # 年率換算の基準は MultiBacktester の calc_metrics とそろえる
        print(portfolio_stats(self.portfolio))
        print("-" * 40)
        print("【最新の注文履歴】")
        # 最新5件を表示
        print(self._orders_readable(self.portfolio).tail(5))
        print("-" * 40)

    def save_orders_to_csv(self, filename: str = "vectorbt_orders.csv"):
        """
        取引履歴をCSV形式で保存します。

        Args:
            filename (str): 保存するファイル名。デフォルトは 'vectorbt_orders.csv'。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return
        # 取引履歴の詳細をCSVとして出力
        self._orders_readable(self.portfolio).to_csv(filename)
        logger.info(f"注文履歴を {filename} に保存しました。")

    def save_chart_to_sheet(self):
        """
        価格、指標、シグナル、売買履歴を結合した一時的なテーブルを作成し、
        スプレッドシートの 'Chart' シートに保存します。
        """
        if (
            self.portfolio is None
            or self.df is None
            or self.entries_series is None
            or self.exits_series is None
        ):
            logger.warning("バックテストが実行されていません。")
            return

        # 1. 注文履歴を日付で集約
        orders = self._orders_readable(self.portfolio).copy()
        # エントリー当日に足を分けた日は、SL/TP の決済が 12:00 になるため、日付にそろえて集計する
        orders["Timestamp"] = pd.to_datetime(orders["Timestamp"]).dt.normalize()

        # 売買代金 (Value) を算出
        orders["Value"] = orders["Size"] * orders["Price"]

        # 同じ日に複数注文がある可能性を考慮して集約
        order_summary = orders.groupby("Timestamp").agg(
            {
                "Size": "sum",
                "Value": "sum",
                "Side": lambda x: ",".join(x.astype(str).unique()),
            }
        )

        # 加重平均価格を算出 (Total Value / Total Size)
        order_summary["Price"] = order_summary["Value"] / order_summary["Size"]

        # カラム名の整理
        order_summary = order_summary.rename(
            columns={
                "Size": "exec_size",
                "Price": "exec_price",
                "Value": "exec_value",
                "Side": "exec_side",
            }
        )[["exec_size", "exec_price", "exec_value", "exec_side"]]

        # 2. データをこの瞬間だけ結合
        export_df = pd.concat(
            [
                self.df,
                self.entries_series.rename("entry_signal"),
                self.exits_series.rename("exit_signal"),
                order_summary,
            ],
            axis=1,
        )

        # 3. スプレッドシートに書き込み
        ds.write_df_to_sheet("Chart", export_df.fillna(""), include_index=True)
        logger.info("売買履歴を含む時系列データを 'Chart' シートに保存しました。")

    def save_orders_to_sheet(self, sheet_name: str = "Orders"):
        """
        詳細な注文履歴をスプレッドシートに保存します。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return

        df = self._orders_readable(self.portfolio)
        ds.write_df_to_sheet(sheet_name, df, include_index=False)
        logger.info(f"注文履歴を '{sheet_name}' シートに保存しました。")

    def save_trades_to_sheet(self, sheet_name: str = "Trades"):
        """
        取引履歴（エントリーからエグジットまでの一連の結果）をスプレッドシートに保存します。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return

        df = self.portfolio.trades.records_readable  # pyright: ignore[reportFunctionMemberAccess]
        ds.write_df_to_sheet(sheet_name, df, include_index=False)
        logger.info(f"取引履歴を '{sheet_name}' シートに保存しました。")

    def show(self):
        """
        バックテストの結果をチャートで表示します。
        """
        if self.portfolio is None:
            logger.warning(
                "バックテストが実行されていません。先に run() を呼び出してください。"
            )
            return

        fig = self.portfolio.plot()
        if fig is None:
            logger.warning("表示できるチャートがありません。")
            return
        fig.show()
