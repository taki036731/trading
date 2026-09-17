from abc import ABC, abstractmethod

import pandas as pd


class AbstractStopLoss(ABC):
    """
    損切り(Stop Loss)ロジックを管理する抽象基底クラス。
    """

    def __init__(self, is_trailing: bool = False, **kwargs):
        """
        AbstractStopLoss を初期化します。

        Args:
            is_trailing (bool): トレイリングストップとして扱う場合は True。デフォルトは False。
            **kwargs: その他のパラメータ。
        """
        self.params = kwargs
        self.is_trailing = is_trailing

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラム名のリストを返します。
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        具体的な損切りしきい値（比率など）を計算します。
        """

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        バリデーションを実行し、各時点の損切りしきい値を計算します。

        Args:
            df (pd.DataFrame): 株価および指標データ。

        Returns:
            pd.Series: vectorbtの `sl_stop` などに渡すための数値Series（通常は比率）。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
