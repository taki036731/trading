import datetime

import pandas as pd
import pandas_ta as ta  # noqa: F401

from lib import setup_logging
from lib.backtester import Backtester
from lib.multi_backtester import MultiBacktester, load_universe
from lib.strategies.exits import (
    AtrStopLoss,
    AtrTakeProfit,
)
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
    InDateRangeSignal,
)

# 検証する銘柄（資金共有モードで同じ日に買いが重なった場合は、この順で先着）
TICKERS = ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T"]

# 検証するパラメータの候補（全組み合わせを実行する）
PARAM_GRID = {
    "short": [5, 10],
    "long": [20, 50],
    "tp_mult": [3, 4],
}


def build_strategy(short: int = 5, long: int = 20, tp_mult: float = 4) -> Backtester:
    """
    strategy01 の戦略を、パラメータを差し替えられる形で作ります。

    Args:
        short (int): 短期 EMA の期間。
        long (int): 長期 EMA の期間。
        tp_mult (float): 利確幅の ATR 倍率。

    Returns:
        Backtester: 新しく作った Backtester。
    """
    return Backtester(
        indicators=[MAIndicator("EMA", short, long), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            InDateRangeSignal(
                datetime.datetime(2016, 6, 25), datetime.datetime(2030, 12, 31)
            ),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(tp_mult),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


if __name__ == "__main__":
    setup_logging()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    data = load_universe(TICKERS, start="2010-01-01")
    mb = MultiBacktester(build_strategy, PARAM_GRID)

    # 1. 銘柄 × パラメータごとに独立して実行
    summary = mb.run_independent(data)
    summary.to_csv("multi_summary.csv", index=False)

    # 2. パラメータごとに銘柄横断で集計（多くの銘柄で安定しているかを見る）
    by_params = mb.aggregate(summary)
    by_params.to_csv("multi_by_params.csv", index=False)
    print("【パラメータごとの銘柄横断集計】")
    print(by_params.sort_values("sharpe_median", ascending=False))

    # 3. 資金共有モード（strategy01 と同じパラメータで、全銘柄を1つの口座で運用）
    portfolio = mb.run_portfolio(data, params={"short": 5, "long": 20, "tp_mult": 4})
    print("-" * 40)
    print("【資金共有モードの結果サマリ】")
    print(portfolio.stats())
    portfolio.trades.records_readable.to_csv("multi_portfolio_trades.csv", index=False)
