import pandas_ta as ta  # noqa: F401

from lib import data_loader as dl
from lib import report
from lib import setup_logging
from lib.backtester import Backtester
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
)

# 評価期間の開始日。データはインジケーターのウォームアップのために、これより前から取得する（#38）
EVAL_START = "2016-06-25"

if __name__ == "__main__":
    setup_logging()
    df = dl.fetch_stock_data("7203.T", start="2010-01-01")

    # バックテスト実行
    backtester = Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )

    portfolio = backtester.run(df, start=EVAL_START)
    report.print_summary(portfolio)
    report.save_orders_csv(portfolio)
    # チャートには、run() が残すインジケーターの列を足した株価データとシグナルを使う
    assert backtester.df is not None
    report.save_chart_to_sheet(
        portfolio,
        backtester.df,
        backtester.entries_series,
        backtester.exits_series,
    )
    report.save_orders_to_sheet(portfolio)
    report.save_trades_to_sheet(portfolio)
    # report.show(portfolio)
    input("Press Enter to close ...")
