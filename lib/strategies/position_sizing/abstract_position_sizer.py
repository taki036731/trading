from abc import ABC, abstractmethod

import pandas as pd
from vectorbt.portfolio.enums import SizeType

from lib.validation import validate_required_columns


class AbstractPositionSizer(ABC):
    """
    売買株数（ポジションサイズ）のロジックを管理する抽象基底クラス。
    """

    # 計算に損切り幅（kwargs の sl_pct）を使うか。True のサイザーは、Backtester で sl を省略できない
    requires_sl_pct: bool = False

    def __init__(
        self,
        size_type: int = SizeType.Amount,
        max_size: float | None = None,
        **kargs,
    ):
        """
        AbstractPositionSizer を初期化します。

        Args:
            size_type (int): vectorbtのSizeType。Amount(0), Value(1), Percent(2)など。
            max_size (float | None): ポジションサイズの上限。単位は size_type に従います
                （Percent なら現金に対する割合、Amount なら株数）。None の場合は上限なし。
            **kargs: その他のパラメータ。

        Raises:
            ValueError: max_size が 0 以下の場合。
        """
        if max_size is not None and max_size <= 0:
            raise ValueError(f"エラー: max_size は正の値にしてください。：{max_size}")
        self.size_type = size_type
        self.max_size = max_size
        self.params = kargs

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        計算に必須なカラム名のリストを返します。

        Returns:
            list[str]: 必須カラム名のリスト。
        """

    @abstractmethod
    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        具体的なサイズ計算ロジックを実装します。

        Args:
            df (pd.DataFrame): 株価データ。
            **kwargs: 計算に必要な追加情報（損切り幅 'sl_pct' など）。

        Returns:
            pd.Series: 各時点のポジションサイズ。
        """

    def generate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        バリデーションを実行し、各エントリ時点でのポジションサイズを生成します。
        max_size があれば、その値で上限を切ります（NaN はそのまま残します）。

        Args:
            df (pd.DataFrame): 株価データ。
            **kwargs: _calculate に渡される追加引数。

        Returns:
            pd.Series: ポジションサイズのSeries（max_size で上限を切ったもの）。

        Raises:
            ValueError: 必須カラムが不足している場合。
        """
        validate_required_columns(df, self.required_columns, self)

        size = self._calculate(df, **kwargs)
        if self.max_size is not None:
            size = size.clip(upper=self.max_size)
        return size
