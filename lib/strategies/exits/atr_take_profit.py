from pandas import DataFrame
from pandas.core.api import Series

from .abstract_take_profit import AbstractTakeProfit


class AtrTakeProfit(AbstractTakeProfit):
    """
    ATR（平均真の範囲）に基づいた利確価格を計算するクラス。
    """

    def __init__(self, percent: float):
        """
        AtrTakeProfit を初期化します。

        Args:
            percent (float): ATRに乗ずる係数（例: 4.0 なら ATRの4倍の幅で利確）。
        """
        super().__init__(percent=percent)
        self.percent = percent

    @property
    def required_columns(self) -> list[str]:
        """
        Returns:
            list[str]: ["atr", "Close"]
        """
        return ["atr", "Close"]

    def _calculate(self, df: DataFrame) -> Series:
        """
        ATRに基づいた利確比率を計算します。

        Returns:
            Series: (atr * percent) / Close
        """
        return (df["atr"] * self.percent) / df["Close"]
