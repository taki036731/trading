from pandas import DataFrame
from pandas.core.api import Series

from .abstract_stop_loss import AbstractStopLoss


class AtrStopLoss(AbstractStopLoss):
    def __init__(self, percent: float, is_trailing: bool = False):
        super().__init__(is_trailing, percent=percent)
        self.percent = percent

    @property
    def required_columns(self) -> list[str]:
        return ["atr", "Close"]

    def _calculate(self, df: DataFrame) -> Series:
        return (df["atr"] * self.percent) / df["Close"]
