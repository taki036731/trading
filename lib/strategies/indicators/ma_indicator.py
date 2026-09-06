from typing import Literal

import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class MAIndicator(AbstractIndicator):
    """
    EMA/SMAを計算するインジケータクラス。
    """

    def __init__(self, method: Literal["EMA", "SMA"], short: int = 20, long: int = 75):
        """
        パラメータの初期化。

        Args:
            method: 計算方法の選択('EMA' or 'SMA')
            short: 短期MAの平均期間
            long: 長期MAの平均期間
        """
        super().__init__(method=method, short=short, long=long)
        self.method = method
        self.short = short
        self.long = long

    @property
    def required_columns(self) -> list[str]:
        """
        インジケータの計算に必要とされるカラム名のリストを定義します。

        Returns:
            list[str]: 必須カラム名のリスト。['Close']
        """
        return ["Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、ATRを計算・列追加して返します。

        Args:
            df (pd.DataFrame): 'Close' カラムを持つ株価データ。

        Returns:
            pd.DataFrame: 元のデータフレームに計算結果('short_ma', 'long_ma')列を追加して返す。
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
