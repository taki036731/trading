from abc import ABC, abstractmethod

import pandas as pd


class AbstractPositionSizer(ABC):
    """
    売買株数（ポジションサイズ）のロジックを管理する抽象クラス
    """

    def __init__(self, size_type: int = 0):  # vbt.portfolio.enums.SizeType.Amount = 0
        """
        Args:
            size_type: vbtのSizeType。Amount(0), Value(1), Percent(2)など
        """
        self.size_type = size_type

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        各時点の売買サイズを計算する
        """
