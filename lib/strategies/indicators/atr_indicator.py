import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class ATRIndicator(AbstractIndicator):
    """
    ATR (Average True Range) を計算するインジケータクラス。
    """

    def __init__(self, period: int = 10, col: str = "atr"):
        """
        ATRIndicator を初期化します。

        Args:
            period (int): ATRの計算期間。デフォルトは 10。
            col (str): 計算結果を入れるカラム名。デフォルトは "atr"。
                ATR を2つ使う場合（例: 損切り用とフィルター用）は、別の名前を指定します。
        """
        super().__init__(period=period, col=col)
        self.period = period
        self.col = col

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必要なカラムを返します。

        Returns:
            list[str]: ["High", "Low", "Close"]
        """
        return ["High", "Low", "Close"]

    @property
    def output_columns(self) -> list[str]:
        """
        追加するカラムを返します。

        Returns:
            list[str]: [col]
        """
        return [self.col]

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATRを計算し、データフレームに col（デフォルトは 'atr'）のカラムを追加します。

        Args:
            df (pd.DataFrame): 'High', 'Low', 'Close' カラムを含むデータフレーム。

        Returns:
            pd.DataFrame: col のカラムが追加されたデータフレーム。
        """
        df[self.col] = df.ta.atr(length=self.period)
        return df
