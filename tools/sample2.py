import datetime

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
    InDateRangeSignal,
)

if __name__ == "__main__":
    setup_logging()
    df = dl.fetch_stock_data("7203.T", start="2010-01-01")

    # バックテスト実行
    backtester = Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            InDateRangeSignal(
                datetime.datetime(2016, 6, 25), datetime.datetime(2030, 12, 31)
            ),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )

    backtester.run(df)
    backtester.display_stats()
    backtester.save_orders_to_csv()
