from typing import Literal

import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class MAIndicator(AbstractIndicator):
    def __init__(self, method: Literal["EMA", "SMA"], short: int = 20, long: int = 75):
        super().__init__(method=method, short=short, long=long)
        self.method = method
        self.short = short
        self.long = long

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、インジケータを計算・列追加して返す。
        子クラス（個別のアルゴリズム）で必ず上書き（オーバーライド）して実装する必要があります。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ

        Returns:
            pd.DataFrame: 元のデータフレームに 'short_ma', 'long_ma' カラムを追加したもの
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
