from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    売買アルゴリズムの基底クラス。
    すべての戦略（Strategy）はこのクラスを継承し、独自のロジックを実装します。
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
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、売買シグナルを判定して返します。
        子クラス（個別のアルゴリズム）で必ず上書き（オーバーライド）して実装する必要があります。

        Args:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' などのカラムを持つ株価データ

        Returns:
            pd.DataFrame: 元のデータフレームに 'Signal' カラムを追加したもの
                          (1: 買いシグナル, -1: 売りシグナル, 0: シグナルなし)
        """
        pass
