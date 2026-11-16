import pandas as pd
from vectorbt.portfolio.enums import SizeType

from .abstract_position_sizer import AbstractPositionSizer


class RiskPercentageSizer(AbstractPositionSizer):
    """
    1トレードあたりの許容リスク（資産のn%）に基づいてポジションサイズを決定するクラス。
    """

    requires_sl_pct = True

    def __init__(self, risk_per_trade: float = 0.02, max_size: float | None = 1.0):
        """
        RiskPercentageSizer を初期化します。

        損切り幅が risk_per_trade より小さいと、サイズが 1.0（現金の100%）を超えます。
        vectorbt は 1.0 を超えた分を黙って現金全額に丸めるため、既定値では 1.0 で上限を切り、
        上限に達したエントリーの件数を Backtester.prepare() が警告します。
        上限に達したエントリーの実際のリスクは、risk_per_trade より小さくなります。

        Args:
            risk_per_trade (float): 1トレードあたりの許容リスク（0.02 = 2%）。
            max_size (float | None): サイズ（現金に対する割合）の上限。デフォルトは 1.0。
        """
        # 資産に対する割合で指定するため SizeType.Percent を使用
        super().__init__(
            size_type=SizeType.Percent, max_size=max_size, risk_per_trade=risk_per_trade
        )
        self.risk_per_trade = risk_per_trade

    @property
    def required_columns(self) -> list[str]:
        """
        Returns:
            list[str]: ["Close"]
        """
        return ["Close"]

    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        リスク許容度と損切り幅からポジションサイズ（資産に対する割合）を計算します。

        Args:
            df (pd.DataFrame): 'Close' カラムを含むデータフレーム。
            **kwargs: 'sl_pct' (pd.Series) が必須。エントリ時点の損切り幅（比率）。

        Returns:
            pd.Series: ポジションサイズ（資産に対する比率）。損切り幅が NaN または 0 以下の日は NaN。

        Raises:
            ValueError: 'sl_pct' が指定されていない場合。
        """
        sl_pct = kwargs.get("sl_pct")
        if sl_pct is None:
            raise ValueError(
                "RiskPercentageSizer._calculate には 'sl_pct' が必要です。"
            )

        # サイズ = 許容リスク / 損切り幅
        # 例: 損切り幅が5% (0.05) でリスク2% (0.02) なら、資産の40% (0.4) を投入
        # 損切り幅がない日（NaN）や 0 以下の日は、リスクを決められないため NaN にして注文を出さない
        # （NaN の割り算は NaN になり、0 で割ると inf になる）
        size = (self.risk_per_trade / sl_pct).where(sl_pct > 0)

        return size
