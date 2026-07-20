# from lib.stock_utils import fetch_stock_data
from lib import data_loader as dl
from lib import setup_logging


def main():
    setup_logging()

    # Appleの株価

    symbol = "AAPL"
    print(f"{symbol} のデータを取得中...")
    df = dl.fetch_stock_data(symbol)
    df = dl.add_sma(df)
    print(df.tail())

    # トヨタの株価
    symbol_jp = "7203.T"
    print(f"\n{symbol_jp} のデータを取得中...")
    df_jp = dl.fetch_stock_data(symbol_jp)
    print(df_jp.tail())


if __name__ == "__main__":
    main()
