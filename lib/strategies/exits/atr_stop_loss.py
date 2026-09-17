from pandas import DataFrame
from pandas.core.api import Series

from .abstract_stop_loss import AbstractStopLoss


class AtrStopLoss(AbstractStopLoss):
    """
    ATR（平均真の範囲）に基づいた損切り価格を計算するクラス。
    """

    def __init__(self, percent: float, is_trailing: bool = False):
        """
        AtrStopLoss を初期化します。

        Args:
            percent (float): ATRに乗ずる係数（例: 2.0 なら ATRの2倍の幅で損切り）。
            is_trailing (bool): トレイリングストップとして扱う場合は True。
        """
        super().__init__(is_trailing, percent=percent)
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
        ATRに基づいた損切り比率を計算します。

        Returns:
            Series: (atr * percent) / Close
        """
        return (df["atr"] * self.percent) / df["Close"]
