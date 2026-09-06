import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class ATRIndicator(AbstractIndicator):
    """
    ATR (Average True Range) を計算するインジケータクラス。
    """

    def __init__(self, period=10):
        """
        ATRIndicator を初期化します。

        Args:
            period (int): ATRの計算期間。デフォルトは 10。
        """
        super().__init__(period=period)
        self.period = period

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
        ATRを計算し、データフレームに 'atr' カラムを追加します。

        Args:
            df (pd.DataFrame): 'Close' カラムを含むデータフレーム。

        Returns:
            pd.DataFrame: 'atr' カラムが追加されたデータフレーム。
        """
        df["atr"] = df.ta.atr(length=self.period)
        return df
