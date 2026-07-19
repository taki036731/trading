import yfinance as yf


def fetch_stock_data(ticker_symbol: str, period: str = "1mo", interval: str = "1d"):
    """
    yfinanceを使用して株価データを取得します。

    Args:
        ticker_symbol (str): ティッカーシンボル (例: 'AAPL', '7203.T')
        period (str): 期間 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): 間隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        pd.DataFrame: 株価データのデータフレーム
    """
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)
    return df


if __name__ == "__main__":
    # 米国株 (例: Apple) の株価を取得
    symbol = "AAPL"
    print(f"{symbol} の直近のデータを取得中...")
    data = fetch_stock_data(symbol)
    print(data.tail())

    # 日本株 (例: トヨタ自動車 7203.T) の場合
    symbol_jp = "7203.T"
    print(f"\n{symbol_jp} の直近のデータを取得中...")
    data_jp = fetch_stock_data(symbol_jp)
    print(data_jp.tail())
