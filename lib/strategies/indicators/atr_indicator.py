import pandas as pd
import pandas_ta as ta  # noqa: F401

from .abstract_indicator import AbstractIndicator


class ATRIndicator(AbstractIndicator):
    """
    ATR (Average True Range) を計算するインジケータクラスです。
    """

    def __init__(self, period=10):
        """
        パラメータの初期化。

        Args:
            period: ATR計算期間
        """
        super().__init__(period=period)
        self.period = period

    @property
    def required_columns(self) -> list[str]:
        """
        インジケータの計算に必要とされるカラム名のリストを定義します。

        Returns:
            list[str]: 必須カラム名のリスト。['Close']
        """
        return ["Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        株価データを受け取り、ATRを計算・列追加して返します。

        Args:
            df (pd.DataFrame): 'Close' カラムを持つ株価データ。

        Returns:
            pd.DataFrame: 元のデータフレームに計算結果('atr')列を追加して返す。
        """
        df["atr"] = df.ta.atr(length=self.period)
        return df
