from pandas import DataFrame
from pandas.core.api import Series

from .abstract_take_profit import AbstractTakeProfit


class AtrTakeProfit(AbstractTakeProfit):
    def __init__(self, percent: float):
        super().__init__(percent=percent)
        self.percent = percent

    @property
    def required_columns(self) -> list[str]:
        return ["atr", "Close"]

    def _calculate(self, df: DataFrame) -> Series:
        return (df["atr"] * self.percent) / df["Close"]
