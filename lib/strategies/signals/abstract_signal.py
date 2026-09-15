from abc import ABC, abstractmethod

import pandas as pd


class AbstractSignal(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        pass

    def generate(self, df: pd.DataFrame) -> pd.Series:
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
