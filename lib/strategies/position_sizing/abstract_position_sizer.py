from abc import ABC, abstractmethod

import pandas as pd
from vectorbt.portfolio.enums import SizeType


class AbstractPositionSizer(ABC):
    """
    売買株数（ポジションサイズ）のロジックを管理する抽象クラス
    """

    def __init__(self, size_type: int = SizeType.Amount, **kargs):
        """
        Args:
            size_type: vbtのSizeType。Amount(0), Value(1), Percent(2)など
        """
        self.size_type = size_type
        self.params = kargs

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        計算に必須な列名のリスト
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        具体的なサイズ計算ロジック
        """

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        バリデーションを実行し、サイズを生成する
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")

        return self._calculate(df, **kwargs)
