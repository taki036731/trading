# from lib.stock_utils import fetch_stock_data
from lib import setup_logging
from lib.DataLoader import DataLoader


def main():
    setup_logging()

    # Appleの株価

    loader = DataLoader("AAPL")
    print(f"{loader.symbol} のデータを取得中...")
    df = loader.fetch_data()
    df = loader.add_sma(df)
    print(df.tail())

    # トヨタの株価
    loader_jp = DataLoader("7203.T")
    print(f"\n{loader_jp.symbol} のデータを取得中...")
    df_jp = loader_jp.fetch_data()
    print(df_jp.tail())


if __name__ == "__main__":
    main()
