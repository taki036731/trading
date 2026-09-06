from abc import ABC, abstractmethod

import pandas as pd


class AbstractIndicator(ABC):
    """
    インジケータ計算の基底クラス。
    DataFrameに対してインジケーター列を追加する。
    """

    def __init__(self, **kwargs):
        """
        アルゴリズムのパラメータを初期化します。

        Args:
            **kwargs: 各戦略固有のパラメータ（例: 短期移動平均の期間、RSIの閾値など）。
                      外部のYAMLファイルなどから渡されることを想定しています。
        """
        self.params = kwargs

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、インジケータを計算・列追加して返す。
        子クラス（個別のアルゴリズム）で必ず上書き（オーバーライド）して実装する必要があります。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ

        Returns:
            pd.DataFrame: 元のデータフレームに 'Signal' カラムを追加したもの
                          (1: 買いシグナル, -1: 売りシグナル, 0: シグナルなし)
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        pass

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return self._calculate(df)
