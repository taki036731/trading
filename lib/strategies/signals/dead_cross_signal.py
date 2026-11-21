import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class DeadCrossSignal(AbstractSignal):
    """
    デッドクロス（短期移動平均が長期移動平均を上から下に突き抜ける）を判定するシグナル。
    """

    def __init__(self, short_col: str = "short_ma", long_col: str = "long_ma"):
        """
        DeadCrossSignal を初期化します。

        Args:
            short_col (str): 短期移動平均のカラム名。デフォルトは "short_ma"。
            long_col (str): 長期移動平均のカラム名。デフォルトは "long_ma"。
        """
        super().__init__(short_col=short_col, long_col=long_col)
        self.short_col = short_col
        self.long_col = long_col

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラムを返します。

        Returns:
            list[str]: [short_col, long_col]
        """
        return [self.short_col, self.long_col]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        デッドクロスを判定します。

        Args:
            df (pd.DataFrame): short_col と long_col のカラムを含むデータフレーム。

        Returns:
            pd.Series: デッドクロスが発生した時点でTrueとなるブール値Series。
        """
        short_ma = df[self.short_col]
        long_ma = df[self.long_col]
        return (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
