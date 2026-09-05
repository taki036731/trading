import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class ATRIndicator(AbstractIndicator):
    def __init__(self, period=10):
        super().__init__(period=period)
        self.period = period

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、インジケータを計算・列追加して返す。
        子クラス（個別のアルゴリズム）で必ず上書き（オーバーライド）して実装する必要があります。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ

        Returns:
            pd.DataFrame: 元のデータフレームに 'atr' カラムを追加したもの
        """
        df["atr"] = df.ta.atr(length=self.period)
        return df
