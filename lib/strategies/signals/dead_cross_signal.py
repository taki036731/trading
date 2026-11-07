import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class DeadCrossSignal(AbstractSignal):
    """
    デッドクロス（短期移動平均が長期移動平均を上から下に突き抜ける）を判定するシグナル。
    """

    def __init__(self):
        """
        DeadCrossSignal を初期化します。
        """
        super().__init__()

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラムを返します。

        Returns:
            list[str]: ["short_ma", "long_ma"]
        """
        return ["short_ma", "long_ma"]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        デッドクロスを判定します。

        Args:
            df (pd.DataFrame): 'short_ma' と 'long_ma' カラムを含むデータフレーム。

        Returns:
            pd.Series: デッドクロスが発生した時点でTrueとなるブール値Series。
        """
        return (df["short_ma"] < df["long_ma"]) & (
            df["short_ma"].shift(1) >= df["long_ma"].shift(1)
        )
