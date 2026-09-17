import datetime as dt

import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class InDateRangeSignal(AbstractSignal):
    """
    指定された日付範囲内であるかを判定するシグナル。
    特定の期間のみトレードを許可する場合などに使用します。
    """

    def __init__(self, begin: dt.datetime, end: dt.datetime):
        """
        InDateRangeSignal を初期化します。

        Args:
            begin (dt.datetime): 開始日時。
            end (dt.datetime): 終了日時。
        """
        super().__init__(begin=begin, end=end)
        self.begin = begin
        self.end = end

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラムを返します。

        Returns:
            list[str]: [] (インデックスの日付のみ使用)
        """
        return []

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        日付が指定範囲内か判定します。

        Args:
            df (pd.DataFrame): インデックスが DatetimeIndex であるデータフレーム。

        Returns:
            pd.Series: 指定範囲内の日付でTrueとなるブール値Series。
        """
        in_date_range = (df.index >= self.begin) & (df.index <= self.end)
        return pd.Series(in_date_range, index=df.index)
