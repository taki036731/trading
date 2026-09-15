import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class ATRConditionSignal(AbstractSignal):
    def __init__(self, lower: float, upper: float):
        super().__init__(lower=lower, upper=upper)
        self.lower = lower
        self.upper = upper

    @property
    def required_columns(self) -> list[str]:
        return ["atr", "Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        df["atr_ratio"] = (df["atr"] / df["Close"]) * 100
        return (df["atr_ratio"] >= self.lower) & (df["atr_ratio"] <= self.upper)
