import pandas as pd
import yfinance as yf

from .abstract_price_source import AbstractPriceSource


class YFinanceSource(AbstractPriceSource):
    """
    yfinance から株価データを取得するクラス。

    Open / High / Low / Close は、auto_adjust=True と同じ調整後価格（株式分割・配当で調整済み）です。
    RawOpen（当時の実際の始値）は、yfinance が返さないため株式分割の履歴から作ります。
    auto_adjust=False でも yfinance の価格は分割調整済みなので、
    その日より後の分割比率をすべて掛けて、分割前の実際の株価に戻します。
    この補正は Yahoo の分割データが正しいことを前提にしています。
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
            df = ticker_obj.history(start=start, auto_adjust=False, actions=True)
        else:
            df = ticker_obj.history(period="max", auto_adjust=False, actions=True)
        if df.empty:
            return df

        # タイムゾーンを消去して日付操作を安定させる
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return to_adjusted_with_raw(df)


def to_adjusted_with_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance の未調整（分割調整のみ）のデータから、調整後の OHLC と実際の始値の列を作ります。

    取得期間が現在まで続いていれば、各日より後の分割はすべて期間内に含まれるため、
    差分取得（期間の途中から取得）でも実際の始値は正しく求まります。

    Args:
        df (pd.DataFrame): auto_adjust=False, actions=True で取得したデータ
            （Open, High, Low, Close, Adj Close, Volume, Stock Splits を含む）。

    Returns:
        pd.DataFrame: Open, High, Low, Close（調整後）, Volume, RawOpen（実際の始値）。
    """
    # 調整後価格は yfinance の auto_adjust と同じく、調整後終値 / 終値 の比率を掛けて作る
    ratio = df["Adj Close"] / df["Close"]
    result = df[["Open", "High", "Low", "Close"]].mul(ratio, axis=0)
    result["Volume"] = df["Volume"]

    # その日より後（当日を含まない）の分割比率の積。分割日の価格はすでに分割後の値になっている
    splits = df["Stock Splits"].replace(0.0, 1.0).fillna(1.0)
    later_splits = splits[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    result["RawOpen"] = df["Open"] * later_splits

    return result
