from abc import ABC, abstractmethod

import pandas as pd


class AbstractTakeProfit(ABC):
    """
    利確(TP)ロジックを管理する抽象クラス
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        pass

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        pass

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        各時点の利確しきい値を計算する

        Returns:
            pd.Series: tp_stop に渡す値のシリーズ
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
