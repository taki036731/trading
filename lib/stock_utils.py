import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker_symbol: str, period: str = "1mo", interval: str = "1d"
) -> pd.DataFrame:
    """yfinanceを使用して株価データを取得します。

    Args:
        ticker_symbol (str): ティッカーシンボル (例: 'AAPL', '7203.T')。
        period (str): 取得期間 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)。デフォルトは '1mo'。
        interval (str): データの間隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)。デフォルトは '1d'。

    Returns:
        pd.DataFrame: 取得した株価データ（Open, High, Low, Close, Volumeなど）を含むデータフレーム。
    """
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)
    return df
