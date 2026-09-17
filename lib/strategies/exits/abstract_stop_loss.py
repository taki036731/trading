from abc import ABC, abstractmethod

import pandas as pd


class AbstractStopLoss(ABC):
    """
    損切り(SL)ロジックを管理する抽象クラス
    """

    def __init__(self, is_trailing: bool = False, **kwargs):
        """
        Args:
            is_trailing (bool): トレイリングストップとして扱う場合は True
        """
        self.params = kwargs
        self.is_trailing = is_trailing

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        pass

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        pass

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        各時点の損切りしきい値を計算する

        Returns:
            pd.Series: sl_stop に渡す値のシリーズ
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
