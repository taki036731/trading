from abc import ABC, abstractmethod

import pandas as pd


class AbstractSignal(ABC):
    """
    売買シグナル判定の抽象基底クラス。
    特定の条件（ゴールデンクロス、期間指定など）に基づいてTrue/FalseのSeriesを生成します。
    """

    def __init__(self, **kwargs):
        """
        シグナルのパラメータを初期化します。

        Args:
            **kwargs: シグナル判定に使用する任意のパラメータ。
        """
        self.params = kwargs

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        具体的なシグナル判定ロジックを実装します。

        Args:
            df (pd.DataFrame): 判定に必要なカラム（指標など）を含むデータフレーム。

        Returns:
            pd.Series: 各行がTrue（条件合致）またはFalse（条件不一致）のブール値Series。
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        シグナル判定に必須なカラム名のリストを定義します。

        Returns:
            list[str]: 必須カラム名のリスト。
        """

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        バリデーションを実行した後、シグナルを生成します。

        Args:
            df (pd.DataFrame): 株価および指標データ。

        Returns:
            pd.Series: 生成されたシグナルのブール値Series。

        Raises:
            ValueError: 必須カラムが不足している場合。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
