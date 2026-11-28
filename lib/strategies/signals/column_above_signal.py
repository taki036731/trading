import pandas as pd

from .abstract_signal import AbstractSignal


class ColumnAboveSignal(AbstractSignal):
    """
    ある列の値が、別の列の値より大きい間、ずっと True になるシグナル。

    クロスした日だけ True になる GoldenCrossSignal と違い、状態を表すので、フィルタとして使います。
    例: 地合いフィルタ（TOPIX の50日線 > 200日線の間だけ買う）
        ColumnAboveSignal("mkt_sma50", "mkt_sma200")
    どちらかの値が NaN の日（指標の計算前の期間など）は False です。
    """

    def __init__(self, left_col: str, right_col: str):
        """
        ColumnAboveSignal を初期化します。

        Args:
            left_col (str): 大きい側であってほしいカラム名。
            right_col (str): 比べる相手のカラム名。
        """
        super().__init__(left_col=left_col, right_col=right_col)
        self.left_col = left_col
        self.right_col = right_col

    @property
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラムを返します。

        Returns:
            list[str]: [left_col, right_col]
        """
        return [self.left_col, self.right_col]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        left_col > right_col を判定します（NaN との比較は False）。

        Args:
            df (pd.DataFrame): left_col と right_col のカラムを含むデータフレーム。

        Returns:
            pd.Series: left_col が right_col より大きい日に True となるブール値Series。
        """
        return df[self.left_col] > df[self.right_col]
