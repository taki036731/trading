from abc import ABC, abstractmethod

import pandas as pd
from vectorbt.portfolio.enums import SizeType


class AbstractPositionSizer(ABC):
    """
    売買株数（ポジションサイズ）のロジックを管理する抽象基底クラス。
    """

    def __init__(self, size_type: int = SizeType.Amount, **kargs):
        """
        AbstractPositionSizer を初期化します。

        Args:
            size_type (int): vectorbtのSizeType。Amount(0), Value(1), Percent(2)など。
            **kargs: その他のパラメータ。
        """
        self.size_type = size_type
        self.params = kargs

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラム名のリストを返します。
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        具体的なサイズ計算ロジックを実装します。

        Args:
            df (pd.DataFrame): 株価データ。
            **kwargs: 計算に必要な追加情報（損切り幅 'sl_pct' など）。
        """

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        バリデーションを実行し、各エントリ時点でのポジションサイズを生成します。

        Args:
            df (pd.DataFrame): 株価データ。
            **kwargs: _calculate に渡される追加引数。

        Returns:
            pd.Series: ポジションサイズのSeries。

        Raises:
            ValueError: 必須カラムが不足している場合。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")

        return self._calculate(df, **kwargs)
