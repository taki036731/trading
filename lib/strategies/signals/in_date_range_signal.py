import datetime as dt

import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class InDateRangeSignal(AbstractSignal):
    def __init__(self, begin: dt.datetime, end: dt.datetime):
        super().__init__(begin=begin, end=end)
        self.begin = begin
        self.end = end

    @property
    def required_columns(self) -> list[str]:
        return []

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        in_date_range = (df.index >= self.begin) & (df.index <= self.end)
        return pd.Series(in_date_range, index=df.index)
