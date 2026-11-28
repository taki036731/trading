import logging

import pandas as pd

from .abstract_indicator import AbstractIndicator

logger = logging.getLogger(__name__)


class ExternalIndicator(AbstractIndicator):
    """
    別の銘柄（TOPIX 連動 ETF など）の株価データで既存のインジケーターを計算し、
    その出力列を、バックテストする銘柄のデータフレームに日付で合わせて追加するインジケーター。

    地合いフィルタのように、売買する銘柄ではなく市場全体の状態で判断するときに使います。
    例: TOPIX（1306.T）の50日線・200日線を、各銘柄の行に mkt_sma50 / mkt_sma200 として追加する。

        ExternalIndicator(
            topix_df,
            MAIndicator("SMA", 50, 200, short_col="mkt_sma50", long_col="mkt_sma200"),
            name="1306.T",
        )

    インジケーターは参照元の全期間で計算してから、日付で合わせます
    （銘柄のデータが短くても、参照元に十分な期間があれば長期の指標が欠けないように）。
    T日の行には参照元のT日の値（T日の終値の時点で分かる値）が入るので、
    これを使うシグナルは、通常のシグナルと同じく1日シフトして使います（未来の情報は入りません）。

    参照元に行がない日（取得元の誤りとして取り除かれた日など）は、前の日の値で埋めます（過去方向のみ）。
    参照元のデータの最終日より後の日は、古い値で埋め続けないように NaN にして警告します。
    参照元の最初の日より前の日も NaN です。
    """

    def __init__(self, source: pd.DataFrame, indicator: AbstractIndicator, name: str = ""):
        """
        ExternalIndicator を初期化します。

        Args:
            source (pd.DataFrame): インジケーターを計算する参照元の株価データ（DatetimeIndex）。
                コピーしてから計算するので、渡したデータフレームは変わりません。
            indicator (AbstractIndicator): 参照元で計算するインジケーター。
                出力列の名前は、売買する銘柄の列とぶつからない名前にします（例: mkt_sma50）。
            name (str): 警告に付ける参照元の名前（ティッカーなど）。
        """
        super().__init__(indicator=indicator, name=name)
        self.source = source
        self.indicator = indicator
        self.name = name

    @property
    def required_columns(self) -> list[str]:
        """
        売買する銘柄のデータフレームに必要なカラムを返します。
        参照元に必要なカラムは、参照元で計算するときに、内側のインジケーターが検証します。

        Returns:
            list[str]: []（インデックスの日付のみ使用）
        """
        return []

    @property
    def output_columns(self) -> list[str]:
        """
        追加するカラムを返します。

        Returns:
            list[str]: 内側のインジケーターの出力カラム。
        """
        return self.indicator.output_columns

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        参照元でインジケーターを計算し、df の日付に合わせて出力列を追加します。

        Args:
            df (pd.DataFrame): 売買する銘柄の株価データ（DatetimeIndex）。

        Returns:
            pd.DataFrame: 出力列を追加したデータフレーム。

        Raises:
            ValueError: 参照元に、内側のインジケーターの必須カラムがない場合。
        """
        source = self.source.copy()
        self.indicator.generate(source)
        values = source[self.output_columns].sort_index()

        # 参照元に行がない日は、前の日の値で埋める（過去方向のみ。後ろの値で埋めると未来の情報になる）
        merged_index = values.index.union(df.index)
        aligned = values.reindex(merged_index).ffill().reindex(df.index)

        # 参照元の最終日より後は、古い値を使い続けないように NaN にする
        last_day = values.index.max() if len(values) else None
        if last_day is not None:
            after_last = df.index > last_day
            if after_last.any():
                prefix = f"[{self.name}] " if self.name else ""
                logger.warning(
                    f"{prefix}参照元のデータが {last_day} で終わっているため、"
                    f"それより後の {int(after_last.sum())} 行の {', '.join(self.output_columns)} を NaN にしました。"
                    "参照元のデータを更新してください。"
                )
                aligned.loc[after_last] = float("nan")

        for col in self.output_columns:
            df[col] = aligned[col]
        return df
