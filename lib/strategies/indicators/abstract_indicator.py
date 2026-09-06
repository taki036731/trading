from abc import ABC, abstractmethod

import pandas as pd


class AbstractIndicator(ABC):
    """
    インジケータ計算の基底クラス。
    DataFrameに対して、テクニカル指標の計算結果の列を追加します。
    """

    def __init__(self, **kwargs):
        """
        アルゴリズムのパラメータを初期化します。

        Args:
            **kwargs: 各戦略固有のパラメータ（例: 短期移動平均の期間、RSIの閾値など）。
        """
        self.params = kwargs

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、インジケータを計算・列追加して返します。
        子クラス（個別のアルゴリズム）で必ずオーバーライド（実装）する必要があります。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ。

        Returns:
            pd.DataFrame: 元のデータフレームに計算結果（'short_ma' カラムなど）を追加したもの。
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        インジケータの計算に必要とされるカラム名のリストを定義します。

        Returns:
            list[str]: 必須カラム名のリスト。
        """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        入力データのバリデーションを行った後、インジケータの計算を実行します。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ。

        Returns:
            pd.DataFrame: 元のデータフレームに計算結果（インジケータやシグナル）を追加したもの。

        Raises:
            ValueError: 必須カラムがデータフレームに含まれていない場合。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
