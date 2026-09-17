import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_signal import AbstractSignal


class ATRConditionSignal(AbstractSignal):
    """
    ATR（平均真の範囲）の価格に対する割合が、指定した範囲内にあるかを判定するシグナル。
    ボラティリティが一定の範囲にある期間を特定するために使用します。
    """

    def __init__(self, lower: float, upper: float):
        """
        ATRConditionSignal を初期化します。

        Args:
            lower (float): ATR比率の最小値（パーセント）。
            upper (float): ATR比率の最大値（パーセント）。
        """
        super().__init__(lower=lower, upper=upper)
        self.lower = lower
        self.upper = upper

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラムを返します。

        Returns:
            list[str]: ["atr", "Close"]
        """
        return ["atr", "Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        ATR比率 (ATR/Close * 100) を計算し、指定範囲内(lower <= ratio <= upper)か判定します。

        Args:
            df (pd.DataFrame): 'atr' と 'Close' カラムを含むデータフレーム。

        Returns:
            pd.Series: 判定結果のブール値Series。
        """
        df["atr_ratio"] = (df["atr"] / df["Close"]) * 100
        return (df["atr_ratio"] >= self.lower) & (df["atr_ratio"] <= self.upper)
