import itertools
import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from lib import data_loader as dl
from lib.backtester import JP_MARKET_PARAMS, Backtester, DateLike, PreparedSignals
from lib.data_sources import AbstractPriceSource
from lib.metrics import (
    METRIC_COLUMNS,
    calc_metrics,
    entry_order_stats,
    warn_low_fill_rate,
)

logger = logging.getLogger(__name__)


def load_universe(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    source: AbstractPriceSource | None = None,
) -> dict[str, pd.DataFrame]:
    """
    複数銘柄の株価データをまとめて取得します。

    銘柄ごとに fetch_stock_data を呼び、データが取得できなかった銘柄（空のデータ、または取得時の例外）は
    警告を出してスキップします。

    Args:
        tickers (Iterable[str]): ティッカーシンボルのリスト（例: ['7203.T', '6758.T']）。
        start (str, optional): データの取得開始日（'YYYY-MM-DD'）。
        end (str, optional): データの取得終了日（'YYYY-MM-DD'）。
        source (AbstractPriceSource, optional): 株価データの取得元。デフォルトは None（YFinanceSource）。

    Returns:
        dict[str, pd.DataFrame]: ティッカーをキー、株価データを値とする辞書（指定順を保持）。
            取得できなかった銘柄は含みません（件数と銘柄は最後にまとめて警告します）。
    """
    data: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for ticker in tickers:
        try:
            df = dl.fetch_stock_data(ticker, start=start, end=end, source=source)
        except Exception as e:  # noqa: BLE001
            # 銘柄が多いと、1銘柄の通信エラーなどで全体が止まらないようにする（理由は警告に出す）
            logger.warning(f"[{ticker}] データの取得に失敗したためスキップします。：{e!r}")
            skipped.append(ticker)
            continue
        if df.empty:
            logger.warning(f"[{ticker}] データが取得できなかったためスキップします。")
            skipped.append(ticker)
            continue
        data[ticker] = df
    # 銘柄が多いと1件ずつの警告を見落としやすいため、最後にまとめて出す
    if skipped:
        logger.warning(
            f"{len(data) + len(skipped)} 銘柄のうち {len(skipped)} 銘柄は"
            f"データが取得できず、スキップしました。：{skipped}"
        )
    return data


def expand_param_grid(param_grid: dict[str, list] | None) -> list[dict[str, Any]]:
    """
    パラメータグリッドを全組み合わせのリストに展開します。

    Args:
        param_grid (dict[str, list] | None): パラメータ名をキー、候補値のリストを値とする辞書。
            None または空の場合は、パラメータなしの1通りだけを返します。

    Returns:
        list[dict[str, Any]]: パラメータの組み合わせのリスト。

    Examples:
        >>> expand_param_grid({"short": [5, 10], "long": [20]})
        [{'short': 5, 'long': 20}, {'short': 10, 'long': 20}]
    """
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]


class MultiBacktester:
    """
    複数銘柄 × パラメータグリッドで、売買戦略を一括検証するクラス。

    戦略は「パラメータを受け取って Backtester を作る関数（factory）」として渡します。
    実行のたびに factory から新しい Backtester を作るため、前の実行の状態は混ざりません。

    - run_independent: 銘柄 × パラメータごとに、それぞれ独立した口座で実行する
    - aggregate: run_independent の結果を、パラメータごとに銘柄横断で集計する
    - run_portfolio: 1つの口座（資金共有）で全銘柄を同時に運用する
    """

    def __init__(
        self,
        factory: Callable[..., Backtester],
        param_grid: dict[str, list] | None = None,
    ):
        """
        MultiBacktester を初期化します。

        Args:
            factory (Callable[..., Backtester]): パラメータをキーワード引数で受け取り、
                新しい Backtester を返す関数。
            param_grid (dict[str, list] | None): パラメータ名と候補値のリスト。
                None の場合は factory を引数なしで1回だけ呼びます。
        """
        self.factory = factory
        self.param_grid = param_grid
        self.param_sets = expand_param_grid(param_grid)

    def run_independent(
        self,
        data: dict[str, pd.DataFrame],
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """
        銘柄 × パラメータの全組み合わせを、それぞれ独立した口座で実行します。

        各実行は Backtester.run と同じ処理なので、単一銘柄で実行した場合と結果は一致します。
        indicator が DataFrame を直接書き換えるため、実行ごとに df のコピーを渡します。
        エントリー注文の約定率が低い実行は、Backtester.run が銘柄・パラメータ付きで警告します。

        Args:
            data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
            start (DateLike | None): 評価期間の開始日（その日を含む）。None の場合は各銘柄のデータの先頭から。
                それより前のデータは、インジケーターのウォームアップにだけ使います（Backtester.run と同じ）。
            end (DateLike | None): 評価期間の終了日（その日を含む）。None の場合は各銘柄のデータの末尾まで。

        Returns:
            pd.DataFrame: 1行1実行の結果。列は ticker、パラメータ名、METRIC_COLUMNS。
                最良の組み合わせだけでなく、全組み合わせを返します。
        """
        rows = []
        for params in self.param_sets:
            for ticker, df in data.items():
                backtester = self.factory(**params)
                portfolio = backtester.run(
                    df, name=f"{ticker} {params}", start=start, end=end
                )
                rows.append({"ticker": ticker, **params, **calc_metrics(portfolio)})
        columns = ["ticker", *self._param_names(), *METRIC_COLUMNS]
        return pd.DataFrame(rows, columns=columns)

    def aggregate(self, summary: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
        """
        run_independent の結果を、パラメータごとに銘柄横断で集計します。

        1銘柄だけ良いパラメータではなく、多くの銘柄で安定して良いパラメータを探すために使います。
        取引が0回の銘柄（1単元を買えず、注文が全て捨てられた銘柄など）は戦略を評価できないため、
        positive_ratio と各指標の中央値・平均からは除外し、件数を zero_trade_tickers に出します。
        決済済みのトレードがなく、保有中のトレードだけの銘柄は取引ありとして扱います
        （total_return などには含み損益が入るため。win_rate などは NaN なので、その指標の集計からは外れます）。

        Args:
            summary (pd.DataFrame): run_independent の戻り値。
            by (list[str] | None): パラメータに加えて集計を分ける列（例: ['sector']、['size']）。
                業種・規模などの列は lib.universe.attach_attributes で summary に結合しておきます。
                None の場合はパラメータだけで分けます。

        Returns:
            pd.DataFrame: パラメータごと（by を指定した場合は、パラメータ × by の値ごと）の集計。
                列はパラメータ名、by の列と、次のとおり。
                - num_tickers: 銘柄数
                - num_traded_tickers: 取引（決済済みか保有中）が1回以上あった銘柄数（集計の対象）
                - zero_trade_tickers: 取引が0回の銘柄数（集計から除外）
                - positive_ratio: total_return がプラスの銘柄の割合（取引があった銘柄のうち）
                - <指標>_median / <指標>_mean: 各指標の銘柄横断の中央値・平均（取引があった銘柄のみ）
                - total_trades: 決済済みのトレード数の合計
                - total_open_trades: 期間の最後に保有中のトレード数の合計
                - fill_rate_median: エントリー注文の約定率の銘柄横断の中央値（全銘柄。注文のない銘柄は除く）
                - total_entry_dropped: 捨てられたエントリー注文の数の合計
        """
        by = list(by or [])
        missing = [col for col in by if col not in summary.columns]
        if missing:
            raise ValueError(f"エラー: 集計を分ける列 {missing} が summary にありません。")
        keys = [*self._param_names(), *by]
        metrics = ["total_return", "cagr", "max_drawdown", "sharpe", "win_rate", "profit_factor"]

        def _agg(group: pd.DataFrame) -> pd.Series:
            traded = group[(group["num_trades"] + group["open_trades"]) > 0]
            result = {
                "num_tickers": len(group),
                "num_traded_tickers": len(traded),
                "zero_trade_tickers": len(group) - len(traded),
                # 取引があった銘柄がなければ NaN（0 にすると「全銘柄マイナス」と区別できない）
                "positive_ratio": (
                    float((traded["total_return"] > 0).mean()) if len(traded) else np.nan
                ),
            }
            for m in metrics:
                # profit_factor は負けトレードがないと inf になるため、集計からは除外する
                values = traded[m].replace([np.inf, -np.inf], np.nan)
                result[f"{m}_median"] = values.median()
                result[f"{m}_mean"] = values.mean()
            result["total_trades"] = int(group["num_trades"].sum())
            result["total_open_trades"] = int(group["open_trades"].sum())
            result["fill_rate_median"] = group["entry_fill_rate"].median()
            result["total_entry_dropped"] = int(group["entry_dropped"].sum())
            return pd.Series(result)

        if not keys:
            return _agg(summary).to_frame().T

        return (
            # by の列に NaN（ユニバースにない銘柄）があっても行を落とさない
            summary.groupby(keys, sort=False, dropna=False)[METRIC_COLUMNS]
            .apply(_agg)
            .reset_index()
        )

    def run_portfolio(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
        init_cash: float | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> vbt.Portfolio:
        """
        1つの口座（資金共有）で全銘柄を同時に運用するバックテストを実行します。

        - 各銘柄で Backtester.prepare を実行し（シフトの扱いは単一銘柄と同じ）、
          列=銘柄の DataFrame にまとめて from_signals を1回呼びます
        - call_seq="auto" により、同じ日は売りを先に処理してから買いを処理します
        - 同じ日に複数銘柄で買いシグナルが出て現金が足りない場合は、サイズの小さい順に先着になります
          （call_seq="auto" は注文の大きさの順に処理するため）。サイズが同じなら data の銘柄の並び順です。
          サイザーの上限（max_size）に達した銘柄同士は同じサイズになるため、data の並び順になります
        - RiskPercentageSizer など SizeType.Percent のサイザーは、資産全体ではなく
          その時点の「現金残高」に対する割合になります
        - 日付は全銘柄の和集合にそろえます。データのない日はエントリー/エグジットを False、
          サイズを NaN にして注文を出しません。終値は時価評価のため前方埋めします（過去方向のみ）
        - 手数料・スリッページ・約定率の警告のしきい値（min_fill_rate）は、factory が作る Backtester の値を使います
        - エントリー注文の約定率が低い銘柄は警告します。資金共有では、1単元に届かない注文に加えて、
          現金が足りない注文（NoCashLong など）も捨てられます
        - エントリー当日に SL/TP ラインへ届く日は、銘柄ごとに足を2行に分けます（split_entry_day_bars）。
          他の銘柄の2行目は、値動きのない足（データのない日と同じ扱い）になります

        Args:
            data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
            params (dict[str, Any] | None): factory に渡すパラメータ。None の場合は引数なし。
            init_cash (float | None): 口座全体の初期資金。None の場合は factory が作る
                Backtester の init_cash を使います。
            start (DateLike | None): 評価期間の開始日（その日を含む）。None の場合は各銘柄のデータの先頭から。
                それより前のデータは、インジケーターのウォームアップにだけ使います（Backtester.run と同じ）。
            end (DateLike | None): 評価期間の終了日（その日を含む）。None の場合は各銘柄のデータの末尾まで。

        Returns:
            vbt.Portfolio: 全銘柄を1グループにまとめた Portfolio。
                stats() は口座全体、trades / orders は銘柄（列）ごとに参照できます。

        Raises:
            ValueError: data が空の場合、銘柄間で size_type が異なる場合、
                または評価期間にデータがない銘柄がある場合。
        """
        if not data:
            raise ValueError("エラー: 銘柄データが空です。")
        params = params or {}

        prepared: dict[str, PreparedSignals] = {}
        cash = init_cash
        cost_params: dict[str, Any] = {}
        min_fill_rate: float | None = None
        for ticker, df in data.items():
            backtester = self.factory(**params)
            if cash is None:
                cash = backtester.init_cash
            cost_params = backtester.cost_params
            min_fill_rate = backtester.min_fill_rate
            # エントリー当日の SL/TP 判定のための足の分割は、Backtester.run と同じく prepare の後に行う
            prepared[ticker] = backtester.prepare(
                df, name=f"資金共有 {params} {ticker}", start=start, end=end
            ).split_entry_day_bars(backtester.slippage)

        size_types = {p.size_type for p in prepared.values()}
        if len(size_types) != 1:
            raise ValueError(f"エラー: 銘柄間で size_type が異なります。：{size_types}")

        combined = self._combine(prepared)
        portfolio = vbt.Portfolio.from_signals(
            **combined,
            init_cash=cash,
            group_by=True,  # 全銘柄を1つの口座としてまとめる
            cash_sharing=True,  # 現金を銘柄間で共有する
            call_seq="auto",  # 同じ日は売り → 買いの順に処理する
            **cost_params,
            **JP_MARKET_PARAMS,
        )
        warn_low_fill_rate(
            entry_order_stats(portfolio), min_fill_rate, label=f"資金共有 {params}"
        )
        return portfolio

    @staticmethod
    def _combine(prepared: dict[str, PreparedSignals]) -> dict:
        """
        銘柄ごとの PreparedSignals を、列=銘柄の DataFrame にまとめます。

        Args:
            prepared (dict[str, PreparedSignals]): ティッカーをキーとする各銘柄の入力。

        Returns:
            dict: from_signals のキーワード引数。キーは PreparedSignals.to_vbt_kwargs() と同じで、
                size_type / sl_trail 以外の値は列=銘柄の DataFrame。
        """

        def _wide(attr: str) -> pd.DataFrame:
            return pd.concat(
                {ticker: getattr(p, attr) for ticker, p in prepared.items()},
                axis=1,
                sort=True,
            ).sort_index()

        close = _wide("close")
        # データのある日（その銘柄が取引された日）
        has_data = close.notna()

        # 時価評価用に、休場・欠損日は直前の終値で埋める（未来方向には埋めない）
        filled_close = close.ffill()

        return {
            "close": filled_close,
            # 欠損日は値動きのない足（始値=高値=安値=直前の終値）として扱い、SL/TP が誤って発動しないようにする
            "open": _wide("open").where(has_data, filled_close),
            "high": _wide("high").where(has_data, filled_close),
            "low": _wide("low").where(has_data, filled_close),
            "price": _wide("price"),
            "entries": _wide("entries").fillna(False).astype(bool) & has_data,
            "exits": _wide("exits").fillna(False).astype(bool) & has_data,
            # 欠損日・データのない期間は NaN（ストップなし）のまま。0 で埋めると建値のストップになる
            "sl_stop": _wide("sl_stop"),
            "tp_stop": _wide("tp_stop"),
            "size": _wide("size").where(has_data),
            "size_type": next(iter(prepared.values())).size_type,
            # 全銘柄が同じパラメータ（同じ factory の Backtester）なので、1つの値で足りる
            "sl_trail": next(iter(prepared.values())).sl_trail,
            "size_granularity": _wide("size_granularity").where(has_data),
            "min_size": _wide("min_size").where(has_data, 0.0),
        }

    def _param_names(self) -> list[str]:
        """
        パラメータ名のリストを返します。

        Returns:
            list[str]: param_grid のキー（未指定なら空リスト）。
        """
        return list(self.param_grid.keys()) if self.param_grid else []
