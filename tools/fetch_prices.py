from lib.stock_utils import fetch_stock_data


def main():
    # Appleの株価
    symbol = "AAPL"
    print(f"{symbol} のデータを取得中...")
    df = fetch_stock_data(symbol)
    print(df.tail())

    # トヨタの株価
    symbol_jp = "7203.T"
    print(f"\n{symbol_jp} のデータを取得中...")
    df_jp = fetch_stock_data(symbol_jp)
    print(df_jp.tail())


if __name__ == "__main__":
    main()
