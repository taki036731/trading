import pandas_ta as ta  # noqa: F401

from lib import data_loader as dl
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

    backtester.run(df, start=EVAL_START)
    backtester.display_stats()
    backtester.save_orders_to_csv()
    backtester.save_chart_to_sheet()
    backtester.save_orders_to_sheet()
    backtester.save_trades_to_sheet()
    # backtester.show()
    input("Press Enter to close ...")
