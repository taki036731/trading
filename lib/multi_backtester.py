import itertools
import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from lib import data_loader as dl
from lib.backtester import JP_MARKET_PARAMS, Backtester, PreparedSignals

logger = logging.getLogger(__name__)

# 独立モードの結果に含める指標の列名
METRIC_COLUMNS = [
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "sortino",
    "num_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "avg_trade_return",
    "exposure",
]


def load_universe(
    tickers: Iterable[str], start: str | None = None, end: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    複数銘柄の株価データをまとめて取得します。

    銘柄ごとに fetch_stock_data を呼び、データが取得できなかった銘柄は警告を出してスキップします。

    Args:
        tickers (Iterable[str]): ティッカーシンボルのリスト（例: ['7203.T', '6758.T']）。
        start (str, optional): データの取得開始日（'YYYY-MM-DD'）。
        end (str, optional): データの取得終了日（'YYYY-MM-DD'）。

    Returns:
        dict[str, pd.DataFrame]: ティッカーをキー、株価データを値とする辞書（指定順を保持）。
    """
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = dl.fetch_stock_data(ticker, start=start, end=end)
        if df.empty:
            logger.warning(f"[{ticker}] データが取得できなかったためスキップします。")
            continue
        data[ticker] = df
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


def _year_freq(index: pd.Index) -> str | None:
    """
    年率換算に使う「1年あたりの本数」を、データの実際の暦日の長さから求めます。

    freq="D" のままだと vectorbt は 1年=365本 として年率換算しますが、
    日本株の日足は営業日だけ（1年あたり約245本）なので、CAGR が小さく、
    シャープレシオが大きく出てしまいます。これを補正するために使います。

    Args:
        index (pd.Index): Portfolio の日付インデックス。

    Returns:
        str | None: vectorbt の year_freq に渡す値（例: '245.1 days'）。
            期間が短すぎて求められない場合は None（vectorbt の既定値を使う）。
    """
    if len(index) < 2:
        return None
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return None
    # freq="D" では1本=1日として扱われるため、「本数/年」日を1年とみなす
    return f"{len(index) / years} days"


def calc_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    """
    Portfolio から、戦略の比較に使う主要な指標を計算します。

    Args:
        portfolio (vbt.Portfolio): 単一銘柄（または資金共有でグループ化済み）の Portfolio。

    Returns:
        dict[str, float]: METRIC_COLUMNS をキーとする指標の辞書。
    """
    trades = portfolio.trades
    num_trades = int(trades.count())
    year_freq = _year_freq(portfolio.wrapper.index)
    return {
        "total_return": float(portfolio.total_return()),
        "cagr": float(portfolio.annualized_return(year_freq=year_freq)),
        "max_drawdown": float(portfolio.max_drawdown()),
        "sharpe": float(portfolio.sharpe_ratio(year_freq=year_freq)),
        "sortino": float(portfolio.sortino_ratio(year_freq=year_freq)),
        "num_trades": num_trades,
        "win_rate": float(trades.win_rate()) if num_trades else np.nan,
        "profit_factor": float(trades.profit_factor()) if num_trades else np.nan,
        "expectancy": float(trades.expectancy()) if num_trades else np.nan,
        "avg_trade_return": float(trades.returns.mean()) if num_trades else np.nan,
        "exposure": float(portfolio.positions.coverage()),
    }


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

    def run_independent(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        銘柄 × パラメータの全組み合わせを、それぞれ独立した口座で実行します。

        各実行は Backtester.run と同じ処理なので、単一銘柄で実行した場合と結果は一致します。
        indicator が DataFrame を直接書き換えるため、実行ごとに df のコピーを渡します。

        Args:
            data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。

        Returns:
            pd.DataFrame: 1行1実行の結果。列は ticker、パラメータ名、METRIC_COLUMNS。
                最良の組み合わせだけでなく、全組み合わせを返します。
        """
        rows = []
        for params in self.param_sets:
            for ticker, df in data.items():
                backtester = self.factory(**params)
                portfolio = backtester.run(df.copy())
                rows.append({"ticker": ticker, **params, **calc_metrics(portfolio)})
        columns = ["ticker", *self._param_names(), *METRIC_COLUMNS]
        return pd.DataFrame(rows, columns=columns)

    def aggregate(self, summary: pd.DataFrame) -> pd.DataFrame:
        """
        run_independent の結果を、パラメータごとに銘柄横断で集計します。

        1銘柄だけ良いパラメータではなく、多くの銘柄で安定して良いパラメータを探すために使います。

        Args:
            summary (pd.DataFrame): run_independent の戻り値。

        Returns:
            pd.DataFrame: パラメータごとの集計。列は次のとおり。
                - num_tickers: 銘柄数
                - positive_ratio: total_return がプラスの銘柄の割合
                - <指標>_median / <指標>_mean: 各指標の銘柄横断の中央値・平均
                - total_trades: 取引回数の合計
        """
        param_names = self._param_names()
        metrics = ["total_return", "cagr", "max_drawdown", "sharpe", "win_rate", "profit_factor"]

        def _agg(group: pd.DataFrame) -> pd.Series:
            result = {
                "num_tickers": len(group),
                "positive_ratio": float((group["total_return"] > 0).mean()),
            }
            for m in metrics:
                # profit_factor は負けトレードがないと inf になるため、集計からは除外する
                values = group[m].replace([np.inf, -np.inf], np.nan)
                result[f"{m}_median"] = values.median()
                result[f"{m}_mean"] = values.mean()
            result["total_trades"] = int(group["num_trades"].sum())
            return pd.Series(result)

        if not param_names:
            return _agg(summary).to_frame().T

        return (
            summary.groupby(param_names, sort=False)[METRIC_COLUMNS]
            .apply(_agg)
            .reset_index()
        )

    def run_portfolio(
        self,
        data: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
        init_cash: float | None = None,
    ) -> vbt.Portfolio:
        """
        1つの口座（資金共有）で全銘柄を同時に運用するバックテストを実行します。

        - 各銘柄で Backtester.prepare を実行し（シフトの扱いは単一銘柄と同じ）、
          列=銘柄の DataFrame にまとめて from_signals を1回呼びます
        - call_seq="auto" により、同じ日は売りを先に処理してから買いを処理します
        - 同じ日に複数銘柄で買いシグナルが出て現金が足りない場合は、data の銘柄の並び順で先着になります
        - RiskPercentageSizer など SizeType.Percent のサイザーは、資産全体ではなく
          その時点の「現金残高」に対する割合になります
        - 日付は全銘柄の和集合にそろえます。データのない日はエントリー/エグジットを False、
          サイズを NaN にして注文を出しません。終値は時価評価のため前方埋めします（過去方向のみ）

        Args:
            data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
            params (dict[str, Any] | None): factory に渡すパラメータ。None の場合は引数なし。
            init_cash (float | None): 口座全体の初期資金。None の場合は factory が作る
                Backtester の init_cash を使います。

        Returns:
            vbt.Portfolio: 全銘柄を1グループにまとめた Portfolio。
                stats() は口座全体、trades / orders は銘柄（列）ごとに参照できます。

        Raises:
            ValueError: data が空の場合、または銘柄間で size_type が異なる場合。
        """
        if not data:
            raise ValueError("エラー: 銘柄データが空です。")
        params = params or {}

        prepared: dict[str, PreparedSignals] = {}
        cash = init_cash
        for ticker, df in data.items():
            backtester = self.factory(**params)
            if cash is None:
                cash = backtester.init_cash
            prepared[ticker] = backtester.prepare(df.copy())

        size_types = {p.size_type for p in prepared.values()}
        if len(size_types) != 1:
            raise ValueError(f"エラー: 銘柄間で size_type が異なります。：{size_types}")

        combined = self._combine(prepared)
        return vbt.Portfolio.from_signals(
            **combined.to_vbt_kwargs(),
            init_cash=cash,
            group_by=True,  # 全銘柄を1つの口座としてまとめる
            cash_sharing=True,  # 現金を銘柄間で共有する
            call_seq="auto",  # 同じ日は売り → 買いの順に処理する
            **JP_MARKET_PARAMS,
        )

    @staticmethod
    def _combine(prepared: dict[str, PreparedSignals]) -> PreparedSignals:
        """
        銘柄ごとの PreparedSignals を、列=銘柄の DataFrame にまとめます。

        Args:
            prepared (dict[str, PreparedSignals]): ティッカーをキーとする各銘柄の入力。

        Returns:
            PreparedSignals: 各属性が列=銘柄の DataFrame になったもの。
        """

        def _wide(attr: str) -> pd.DataFrame:
            return pd.concat(
                {ticker: getattr(p, attr) for ticker, p in prepared.items()}, axis=1
            ).sort_index()

        close = _wide("close")
        # データのある日（その銘柄が取引された日）
        has_data = close.notna()

        return PreparedSignals(
            # 時価評価用に、休場・欠損日は直前の終値で埋める（未来方向には埋めない）
            close=close.ffill(),
            price=_wide("price"),
            entries=_wide("entries").fillna(False).astype(bool) & has_data,
            exits=_wide("exits").fillna(False).astype(bool) & has_data,
            sl_stop=_wide("sl_stop").fillna(0.0),
            tp_stop=_wide("tp_stop").fillna(0.0),
            size=_wide("size").where(has_data),
            size_type=next(iter(prepared.values())).size_type,
        )

    def _param_names(self) -> list[str]:
        """
        パラメータ名のリストを返します。

        Returns:
            list[str]: param_grid のキー（未指定なら空リスト）。
        """
        return list(self.param_grid.keys()) if self.param_grid else []
