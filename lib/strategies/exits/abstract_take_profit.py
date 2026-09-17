from abc import ABC, abstractmethod

import pandas as pd


class AbstractTakeProfit(ABC):
    """
    利確(Take Profit)ロジックを管理する抽象基底クラス。
    """

    def __init__(self, **kwargs):
        """
        AbstractTakeProfit を初期化します。

        Args:
            **kwargs: その他のパラメータ。
        """
        self.params = kwargs

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラム名のリストを返します。
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        具体的な利確しきい値（比率など）を計算します。
        """

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        バリデーションを実行し、各時点の利確しきい値を計算します。

        Args:
            df (pd.DataFrame): 株価および指標データ。

        Returns:
            pd.Series: vectorbtの `tp_stop` などに渡すための数値Series（通常は比率）。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
