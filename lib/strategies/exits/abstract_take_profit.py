from abc import ABC, abstractmethod

import pandas as pd

from lib.validation import validate_required_columns


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

        Returns:
            list[str]: 必須カラム名のリスト。
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        具体的な利確しきい値（比率など）を計算します。

        Args:
            df (pd.DataFrame): 指標等を含むデータフレーム。

        Returns:
            pd.Series: 各時点の利確幅（Closeに対する比率等）。
        """

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        バリデーションを実行し、各時点の利確しきい値を計算します。

        Args:
            df (pd.DataFrame): 株価および指標データ。

        Returns:
            pd.Series: vectorbtの `tp_stop` などに渡すための数値Series（通常は比率）。
        """
        validate_required_columns(df, self.required_columns, self)
        return self._calculate(df)
