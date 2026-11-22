import logging
from typing import cast

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import OrderStatus, OrderStatusInfo
from vectorbt.portfolio.logs import Logs
from vectorbt.portfolio.trades import Positions, Trades

logger = logging.getLogger(__name__)

# エントリー注文の約定率がこれを下回ったら警告する（Backtester の引数 min_fill_rate で変えられる）
DEFAULT_MIN_FILL_RATE = 0.8

# 独立モードの結果に含める指標の列名
METRIC_COLUMNS = [
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "sortino",
    "num_trades",
    "open_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "avg_trade_return",
    "exposure",
    "entry_orders",
    "entry_dropped",
    "entry_fill_rate",
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


def entry_order_stats(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """
    エントリー（買い）注文のうち、約定した数と捨てられた数を、列（銘柄）ごとに集計します。

    サイズが1単元に届かない注文などは、vectorbt がエラーもなく捨てるため、取引履歴には残りません。
    注文の記録（from_signals の log=True。JP_MARKET_PARAMS で指定済み）から数えます。
    保有中に出たエントリーシグナルは注文にならないため、数に含みません。

    Args:
        portfolio (vbt.Portfolio): log=True で実行した Portfolio。

    Returns:
        pd.DataFrame: 列（銘柄）をインデックスとする集計。列は次のとおり。
            - entry_orders: 出したエントリー注文の数
            - entry_filled: そのうち約定した数（一部約定を含む）
            - entry_dropped: そのうち捨てられた数
            - entry_fill_rate: entry_filled / entry_orders（注文がなければ NaN）
            - dropped_<理由>: 捨てられた理由（OrderStatusInfo の名前）ごとの数。出た理由の列だけ
    """
    # vectorbt の cached_property を Pylance が関数と見なすため、実際の型を明示する
    records = cast(Logs, portfolio.logs).values
    columns = portfolio.wrapper.columns
    # longonly なので、買い注文は要求サイズが正のもの（売りは負）
    buys = records[records["req_size"] > 0]
    filled = buys["res_status"] == OrderStatus.Filled

    result = pd.DataFrame(
        {
            "entry_orders": np.bincount(buys["col"], minlength=len(columns)),
            "entry_filled": np.bincount(buys["col"][filled], minlength=len(columns)),
        },
        index=columns,
    )
    result["entry_dropped"] = result["entry_orders"] - result["entry_filled"]
    result["entry_fill_rate"] = (
        result["entry_filled"] / result["entry_orders"].replace(0, np.nan)
    )

    dropped = buys[~filled]
    for info in np.unique(dropped["res_status_info"]):
        name = OrderStatusInfo._fields[int(info)]
        result[f"dropped_{name}"] = np.bincount(
            dropped["col"][dropped["res_status_info"] == info], minlength=len(columns)
        )
    return result


def warn_low_fill_rate(
    stats: pd.DataFrame, min_fill_rate: float | None, label: str = ""
) -> None:
    """
    エントリー注文の約定率が min_fill_rate を下回った列（銘柄）について、警告を出します。

    Args:
        stats (pd.DataFrame): entry_order_stats の戻り値。
        min_fill_rate (float | None): 警告のしきい値（0〜1）。None の場合は何もしません。
        label (str): 警告の先頭に付ける名前（銘柄・パラメータなど）。
    """
    if min_fill_rate is None:
        return
    reason_cols = [c for c in stats.columns if c.startswith("dropped_")]
    for col, row in stats.iterrows():
        rate = row["entry_fill_rate"]
        if pd.isna(rate) or rate >= min_fill_rate:
            continue
        reasons = ", ".join(
            f"{c.removeprefix('dropped_')}={int(row[c])}" for c in reason_cols if row[c]
        )
        name = " ".join(str(x) for x in (label, col if len(stats) > 1 else "") if x)
        logger.warning(
            f"[{name}] エントリー注文の約定率が低くなっています: "
            f"{rate:.0%}（{int(row['entry_filled'])}/{int(row['entry_orders'])}件、"
            f"捨てられた理由: {reasons}）"
        )


def closed_and_open_trades(portfolio: vbt.Portfolio) -> tuple[Trades, Trades]:
    """
    Portfolio のトレードを、決済済みと、期間の最後に保有中（未決済）に分けて返します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。

    Returns:
        tuple[Trades, Trades]: (決済済みのトレード, 保有中のトレード)。
    """
    # vectorbt の cached_property を Pylance が関数と見なすため、実際の型を明示する
    trades = cast(Trades, portfolio.trades)
    # closed / open は、vectorbt がクラスの定義時に追加する属性で、Pylance からは見えない
    closed = cast(Trades, trades.closed)  # pyright: ignore[reportAttributeAccessIssue]
    open_ = cast(Trades, trades.open)  # pyright: ignore[reportAttributeAccessIssue]
    return closed, open_


def calc_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    """
    Portfolio から、戦略の比較に使う主要な指標を計算します。

    トレード単位の指標（num_trades / win_rate / profit_factor / expectancy / avg_trade_return）は、
    決済済みのトレードだけで計算します。期間の最後に保有中のトレードは含み損益で勝ち・負けが決まり、
    データの終了日を変えるだけで値が変わるためです（portfolio_stats の Win Rate なども決済済みだけ）。
    保有中のトレードの件数は open_trades に出します。
    total_return / cagr / sharpe / exposure などの口座単位の指標は、含み損益を含みます。

    Args:
        portfolio (vbt.Portfolio): 単一銘柄（または資金共有でグループ化済み）の Portfolio。

    Returns:
        dict[str, float]: METRIC_COLUMNS をキーとする指標の辞書。
    """
    trades, open_trades = closed_and_open_trades(portfolio)
    # vectorbt の cached_property を Pylance が関数と見なすため、実際の型を明示する
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
        "open_trades": int(open_trades.count()),
        "win_rate": float(trades.win_rate()) if num_trades else np.nan,
        "profit_factor": float(trades.profit_factor()) if num_trades else np.nan,
        "expectancy": float(trades.expectancy()) if num_trades else np.nan,
        "avg_trade_return": float(trades.returns.mean()) if num_trades else np.nan,  # pyright: ignore[reportAttributeAccessIssue]
        "exposure": float(positions.coverage()),
        **_entry_metrics(portfolio),
    }


def _entry_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    """
    calc_metrics 用に、エントリー注文の集計を全列の合計で返します。

    Args:
        portfolio (vbt.Portfolio): log=True で実行した Portfolio。

    Returns:
        dict[str, float]: entry_orders / entry_dropped / entry_fill_rate。
    """
    stats = entry_order_stats(portfolio)
    orders = int(stats["entry_orders"].sum())
    filled = int(stats["entry_filled"].sum())
    return {
        "entry_orders": orders,
        "entry_dropped": orders - filled,
        "entry_fill_rate": filled / orders if orders else np.nan,
    }
