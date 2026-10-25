from abc import ABC, abstractmethod

import pandas as pd


class AbstractPriceSource(ABC):
    """
    株価データの取得元の基底クラス。

    yfinance や J-Quants API などの取得元ごとの違いを吸収し、
    バックテストで使う列をそろえた DataFrame を返します。
    取得元ごとに必要な価格の補正（調整後価格の作り方など）も、このクラスの子クラスが担います。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        取得元の名前を返します。キャッシュの保存先のフォルダ名に使います。

        Returns:
            str: 取得元の名前（例: 'yfinance'）。
        """

    @property
    def required_columns(self) -> list[str]:
        """
        取得したデータに必須なカラム名のリストを返します。

        Open / High / Low / Close は、株式分割・配当で調整した価格です（リターンの計算に使う）。
        RawOpen は、調整していない当時の実際の始値です（100株単元の判定に使う）。

        Returns:
            list[str]: ["Open", "High", "Low", "Close", "Volume", "RawOpen"]
        """
        return ["Open", "High", "Low", "Close", "Volume", "RawOpen"]

    @abstractmethod
    def _fetch(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        """
        取得元から日足の株価データを取得します。

        Args:
            ticker (str): 銘柄のティッカーシンボル。
            start (str, optional): データの取得開始日（'YYYY-MM-DD'）。None の場合は全期間。

        Returns:
            pd.DataFrame: タイムゾーンなしの日付インデックスを持つ株価データ。
                データが存在しない場合は空のデータフレーム。
        """

    def fetch(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        """
        株価データを取得し、必須カラムがそろっているかを検証します。

        Args:
            ticker (str): 銘柄のティッカーシンボル。
            start (str, optional): データの取得開始日（'YYYY-MM-DD'）。None の場合は全期間。

        Returns:
            pd.DataFrame: 株価データ。データが存在しない場合は空のデータフレーム。

        Raises:
            ValueError: 取得したデータに必須カラムが不足している場合。
        """
        df = self._fetch(ticker, start)
        if df.empty:
            return df

        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        return df
