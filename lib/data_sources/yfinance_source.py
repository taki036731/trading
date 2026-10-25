import pandas as pd
import yfinance as yf

from .abstract_price_source import AbstractPriceSource


class YFinanceSource(AbstractPriceSource):
    """
    yfinance から株価データを取得するクラス。

    価格は auto_adjust=True の調整後価格（株式分割・配当で調整済み）を使います。
    """

    @property
    def name(self) -> str:
        """
        Returns:
            str: 'yfinance'
        """
        return "yfinance"

    def _fetch(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        """
        yfinanceを使用してヒストリカルデータを取得します。

        Args:
            ticker (str): 取得する銘柄のティッカーシンボル。
            start (str, optional): データの取得開始日（'YYYY-MM-DD'）。
                指定がない場合は全期間（period="max"）を取得します。デフォルトは None。

        Returns:
            pd.DataFrame: 取得した株価データ（Open, High, Low, Close, Volume）。
                タイムゾーン情報は削除されます。
        """
        ticker_obj = yf.Ticker(ticker)
        if start is not None:
            df = ticker_obj.history(start=start, auto_adjust=True)
        else:
            df = ticker_obj.history(period="max", auto_adjust=True)

        # タイムゾーンを消去して日付操作を安定させる
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        return df
