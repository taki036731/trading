import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class GoldenCrossSignal(AbstractSignal):
    def __init__(self):
        super().__init__()

    @property
    def required_columns(self) -> list[str]:
        return ["short_ma", "long_ma"]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        return (df["short_ma"] > df["long_ma"]) & (
            df["short_ma"].shift(1) <= df["long_ma"].shift(1)
        )
