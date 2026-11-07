from typing import cast

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.trades import Positions, Trades

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


def year_freq(index: pd.Index) -> str | None:
    """
    年率換算に使う「1年あたりの本数」を、データの実際の暦日の長さから求めます。

    freq="D" のままだと vectorbt は 1年=365本 として年率換算しますが、
    日本株の日足は営業日だけ（1年あたり約245本）なので、CAGR が小さく、
    シャープレシオが大きく出てしまいます。これを補正するために使います。
    単一銘柄（Backtester）と複数銘柄（MultiBacktester）で、同じ基準で年率換算するために共通化しています。

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
    # freq="D" では1本=1日として扱われるため、「本数/年」日を1年とみなす。
    # エントリー当日に足を分けた行も本数に含める。vectorbt は行数で年率換算するため、
    # 含めておけば CAGR は暦日の長さどおりになる
    return f"{len(index) / years} days"


def portfolio_stats(portfolio: vbt.Portfolio) -> pd.Series:
    """
    year_freq で年率換算した、Portfolio の統計サマリを返します。

    portfolio.stats() をそのまま呼ぶと 1年=365本 で年率換算されるため、
    calc_metrics とシャープレシオなどが食い違います。表示にはこちらを使います。

    Args:
        portfolio (vbt.Portfolio): 単一銘柄（または資金共有でグループ化済み）の Portfolio。

    Returns:
        pd.Series: portfolio.stats() と同じ形式の統計サマリ。
    """
    freq = year_freq(portfolio.wrapper.index)
    if freq is None:
        return cast(pd.Series, portfolio.stats())
    return cast(pd.Series, portfolio.stats(settings=dict(year_freq=freq)))


def calc_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    """
    Portfolio から、戦略の比較に使う主要な指標を計算します。

    Args:
        portfolio (vbt.Portfolio): 単一銘柄（または資金共有でグループ化済み）の Portfolio。

    Returns:
        dict[str, float]: METRIC_COLUMNS をキーとする指標の辞書。
    """
    # vectorbt の cached_property を Pylance が関数と見なすため、実際の型を明示する
    trades = cast(Trades, portfolio.trades)
    positions = cast(Positions, portfolio.positions)
    num_trades = int(trades.count())
    freq = year_freq(portfolio.wrapper.index)
    # annualized_return や trades.returns などは、vectorbt がクラスの定義時に追加する属性で、Pylance からは見えない
    return {
        "total_return": float(portfolio.total_return()),
        "cagr": float(portfolio.annualized_return(year_freq=freq)),  # pyright: ignore[reportAttributeAccessIssue]
        "max_drawdown": float(portfolio.max_drawdown()),  # pyright: ignore[reportAttributeAccessIssue]
        "sharpe": float(portfolio.sharpe_ratio(year_freq=freq)),  # pyright: ignore[reportAttributeAccessIssue]
        "sortino": float(portfolio.sortino_ratio(year_freq=freq)),  # pyright: ignore[reportAttributeAccessIssue]
        "num_trades": num_trades,
        "win_rate": float(trades.win_rate()) if num_trades else np.nan,
        "profit_factor": float(trades.profit_factor()) if num_trades else np.nan,
        "expectancy": float(trades.expectancy()) if num_trades else np.nan,
        "avg_trade_return": float(trades.returns.mean()) if num_trades else np.nan,  # pyright: ignore[reportAttributeAccessIssue]
        "exposure": float(positions.coverage()),
    }
