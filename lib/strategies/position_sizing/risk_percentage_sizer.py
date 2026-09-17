import pandas as pd
from vectorbt.portfolio.enums import SizeType

from .abstract_position_sizer import AbstractPositionSizer


class RiskPercentageSizer(AbstractPositionSizer):
    """
    1トレードあたりの許容リスク（資産のn%）に基づいてポジションサイズを決定するクラス
    """

    def __init__(self, risk_per_trade: float = 0.02):
        """
        Args:
            risk_per_trade: 1トレードあたりの許容リスク（0.02 = 2%）
        """
        # 資産に対する割合で指定するため SizeType.Percent を使用
        super().__init__(size_type=SizeType.Percent, risk_per_trade=risk_per_trade)
        self.risk_per_trade = risk_per_trade

    @property
    def required_columns(self) -> list[str]:
        return ["Close"]

    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Args:
            df: DataFrame
            **kwargs: sl_pct (pd.Series) が必須
        """
        sl_pct = kwargs.get("sl_pct")
        if sl_pct is None:
            raise ValueError(
                "RiskPercentageSizer._calculate には 'sl_pct' が必要です。"
            )

        # サイズ = 許容リスク / 損切り幅
        # 例: 損切り幅が5% (0.05) でリスク2% (0.02) なら、資産の40% (0.4) を投入
        size = self.risk_per_trade / sl_pct

        return size
