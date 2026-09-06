from typing import Literal

import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class MAIndicator(AbstractIndicator):
    """
    EMA (指数平滑移動平均) または SMA (単純移動平均) を計算するインジケータクラス。
    短期と長期の2本の移動平均線を計算します。
    """

    def __init__(self, method: Literal["EMA", "SMA"], short: int = 20, long: int = 75):
        """
        MAIndicator を初期化します。

        Args:
            method (Literal["EMA", "SMA"]): 計算方法 ('EMA' または 'SMA')。
            short (int): 短期移動平均の期間。デフォルトは 20。
            long (int): 長期移動平均の期間。デフォルトは 75。
        """
        super().__init__(method=method, short=short, long=long)
        self.method = method
        self.short = short
        self.long = long

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必要なカラムを返します。

        Returns:
            list[str]: ["Close"]
        """
        return ["Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移動平均を計算し、データフレームに 'short_ma' と 'long_ma' カラムを追加します。

        Args:
            df (pd.DataFrame): 'Close' カラムを含むデータフレーム。

        Returns:
            pd.DataFrame: 'short_ma' と 'long_ma' カラムが追加されたデータフレーム。

        Raises:
            ValueError: 未対応の method が指定された場合。
        """
        if self.method == "EMA":
            df["short_ma"] = df.ta.ema(length=self.short)
            df["long_ma"] = df.ta.ema(length=self.long)
        elif self.method == "SMA":
            df["short_ma"] = df.ta.sma(length=self.short)
            df["long_ma"] = df.ta.sma(length=self.long)
        else:
            raise ValueError(f"未対応のインジケータです。：{self.method}")
        return df
