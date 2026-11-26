"""
strategy01 の固定利確（TP）と、トレーリングストップ（TP なし）を、複数銘柄で比べる実験用のスクリプト。

比べる設定（VARIANTS）:
- fixed_tp: strategy01 と同じ（SL = ATR×2 の固定、TP = ATR×4）
- fixed_sl_no_tp: SL = ATR×2 の固定、TP なし（「TP を外した効果」と「トレーリングの効果」を分けるための対照）
- trail_<k>: SL = ATR×k のトレーリング、TP なし

エグジットのシグナル（デッドクロス）は全ての設定で残します。
RiskPercentageSizer は損切り幅でサイズを決めるため、トレーリングの幅（k）を変えると、
1トレードのリスクを risk_per_trade にそろえたまま、ポジションサイズも 1/k で変わります。

実行: python -m tools.compare_trailing
出力: compare_trailing_summary.csv（銘柄 × 設定）、compare_trailing_by_variant.csv（設定ごとの集計）、
     compare_trailing_portfolio.csv（資金共有モードの設定ごとの結果）
"""

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401

from lib import setup_logging
from lib.backtester import Backtester
from lib.metrics import calc_metrics
from lib.multi_backtester import MultiBacktester, load_universe
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
)
from lib.universe import load_universe_file, universe_path

logger = logging.getLogger("tools.compare_trailing")

UNIVERSE_NAME = "jpx_stratified_100"
EVAL_START = "2016-06-25"
INIT_CASH = 10_000_000.0

# 設定名 -> (SL の ATR 倍率, トレーリングか, TP の ATR 倍率。None なら TP なし)
VARIANTS: dict[str, tuple[float, bool, float | None]] = {
    "fixed_tp": (2, False, 4),
    "fixed_sl_no_tp": (2, False, None),
    "trail_2": (2, True, None),
    "trail_3": (3, True, None),
    "trail_4": (4, True, None),
}

# 表示する指標（資金共有モードの表の列）
SHOW_METRICS = [
    "cagr",
    "max_drawdown",
    "sharpe",
    "win_rate",
    "profit_factor",
    "num_trades",
    "avg_trade_return",
    "exposure",
]
# 独立モードの集計で表示する列（aggregate の列 + このスクリプトで足す中央値）
SHOW_AGG = [
    "variant",
    "positive_ratio",
    "cagr_median",
    "max_drawdown_median",
    "sharpe_median",
    "win_rate_median",
    "profit_factor_median",
    "total_trades",
    "avg_trade_return_median",
    "exposure_median",
]


def build_strategy(variant: str = "fixed_tp") -> Backtester:
    """
    VARIANTS の設定で strategy01 の Backtester を作ります。

    Args:
        variant (str): VARIANTS のキー。

    Returns:
        Backtester: 新しく作った Backtester。
    """
    sl_mult, is_trailing, tp_mult = VARIANTS[variant]
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(tp_mult) if tp_mult is not None else None,
        sl=AtrStopLoss(sl_mult, is_trailing=is_trailing),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=INIT_CASH,
    )


def compare(data: dict[str, pd.DataFrame], start: str | None = EVAL_START) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    全ての設定を、独立モード（銘柄ごとの口座）と資金共有モード（1つの口座）で実行します。

    Args:
        data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
        start (str | None): 評価期間の開始日。

    Returns:
        tuple: (銘柄 × 設定の結果, 設定ごとの銘柄横断集計, 資金共有モードの設定ごとの結果)
    """
    mb = MultiBacktester(build_strategy, {"variant": list(VARIANTS)})
    summary = mb.run_independent(data, start=start)
    by_variant = mb.aggregate(summary)
    # aggregate にない指標の中央値を足す（aggregate と同じく、取引0回の銘柄は除く）
    traded = summary[(summary["num_trades"] + summary["open_trades"]) > 0]
    extra = traded.groupby("variant", sort=False)[["avg_trade_return", "exposure"]].median()
    by_variant = by_variant.merge(
        extra.add_suffix("_median"), left_on="variant", right_index=True, how="left"
    )
    rows = {}
    for variant in VARIANTS:
        portfolio = mb.run_portfolio(data, params={"variant": variant}, start=start)
        rows[variant] = calc_metrics(portfolio)
    portfolio_table = pd.DataFrame.from_dict(rows, orient="index")
    return summary, by_variant, portfolio_table


if __name__ == "__main__":
    setup_logging()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    universe = load_universe_file(universe_path(UNIVERSE_NAME))
    data = load_universe(universe["ticker"], start="2010-01-01")
    summary, by_variant, portfolio_table = compare(data)

    summary.to_csv("compare_trailing_summary.csv", index=False)
    by_variant.to_csv("compare_trailing_by_variant.csv", index=False)
    portfolio_table.to_csv("compare_trailing_portfolio.csv")

    print(f"【独立モード: 銘柄横断の中央値（{UNIVERSE_NAME}, {len(data)}銘柄）】")
    print(by_variant[SHOW_AGG].to_string(index=False))
    print("\n【資金共有モード: 全銘柄を1つの口座で運用】")
    print(portfolio_table[[*SHOW_METRICS, "open_trades", "entry_fill_rate"]].to_string())
