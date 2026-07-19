import logging
import os

import pandas as pd
import yfinance as yf


class DataLoader:
    """株価データの取得と前処理を行うクラス"""

    CACHE_DIR = "data"
    logger = logging.getLogger(__name__)

    def __init__(self, symbol: str):
        """
        Args:
            symbol (str): 取得する銘柄のティッカーシンボル（例: '7203.T', 'AAPL'）。日本株の場合は '7974.T' のように .T を付けるのが一般的です
        """
        self.__symbol = symbol
        self.__ticker = yf.Ticker(symbol)  # インスタンスを保持

    @property
    def symbol(self):
        return self.__symbol

    def fetch_data(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """指定された銘柄のヒストリカルデータを取得し、Parquet形式でキャッシュ・更新を行う関数。

        ローカルの `data` ディレクトリにParquetファイルが存在する場合はそれを読み込みます。
        キャッシュが存在する場合でも、データ内の最新日付を確認し、不足している新しいデータがあれば
        yfinanceから自動的に取得してキャッシュに追記（差分更新）します。

        Args:
            start (str, optional): データの取得開始日（'YYYY-MM-DD'）。デフォルトは None（全期間）。
            end (str, optional): データの取得終了日（'YYYY-MM-DD'）。デフォルトは None（最新まで）。

        Returns:
            pd.DataFrame: 指定された期間の株価データを含むデータフレーム。
                        データの取得に失敗した場合や存在しない場合は空のデータフレームを返します。
        """
        os.makedirs(DataLoader.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(DataLoader.CACHE_DIR, f"{self.__symbol}.parquet")

        # 1. データの用意（キャッシュの読み込みと差分更新、または全期間の新規取得）
        if os.path.exists(cache_path):
            DataLoader.logger.info(
                f"[{self.__symbol}] キャッシュからデータを読み込みます。"
            )
            df = pd.read_parquet(cache_path, engine="auto")

            # タイムゾーンを消去して日付操作を安定させる
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # キャッシュの最新日付を取得
            last_date = df.index.max()
            # yfinanceへのリクエストは self.ticker を使用
            df_new = self.__ticker.history(start=last_date + pd.Timedelta(days=1))
            df_new = df_new[["Open", "High", "Low", "Close", "Volume"]]
            if not df_new.empty:
                if (
                    isinstance(df_new.index, pd.DatetimeIndex)
                    and df_new.index.tz is not None
                ):
                    df_new.index = df_new.index.tz_localize(None)
                DataLoader.logger.info(
                    f"[{self.__symbol}] {last_date + pd.Timedelta(days=1)} 以降の新しいデータを取得し、キャッシュを更新します。"
                )
                # 既存データと新規データを結合
                df = pd.concat([df, df_new])
                # 重複データが存在する場合は排除（念のための安全策）し、日付順にソート
                df = df[~df.index.duplicated(keep="last")].sort_index()
                # 更新したデータセットで上書き保存
                df.to_parquet(cache_path, engine="auto")
            else:
                DataLoader.logger.info(
                    f"[{self.__symbol}] 追加すべき新しいデータはありません（キャッシュは最新です）。"
                )

        else:
            DataLoader.logger.info(
                f"[{self.__symbol}] キャッシュが存在しないため、yfinanceから全期間取得します。"
            )
            # キャッシュ構築のため、最初は全期間 (period="max") を取得
            df = self.__ticker.history(period="max")
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            if not df.empty:
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_parquet(cache_path, engine="auto")
                DataLoader.logger.info(
                    f"[{self.__symbol}] 全期間データをキャッシュに保存しました。"
                )
            else:
                DataLoader.logger.warning(
                    f"[{self.__symbol}] データの取得に失敗したか、データが存在しません。"
                )
                return df

        # 2. 指定された期間でデータを絞り込んで返す
        # 日付文字列を Timestamp に変換して比較を確実にする
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def add_sma(
        self, df: pd.DataFrame, window: int = 20, column_name: str | None = None
    ) -> pd.DataFrame:
        """
        DataFrameに単純移動平均線(SMA)を追加します

        Args:
            df (pd.DataFrame): fetch_dataで取得した株価データ
            window (int): 移動平均の期間
            column_name (str): 追加する列の名前。指定がない場合は 'SMA_20' のようになります
        """
        if column_name is None:
            column_name = f"SMA_{window}"

        # 元のデータを壊さないようコピーを作成
        df = df.copy()
        # pandasのrollingを使ってSMAを計算し、新しい列として追加
        df[column_name] = df["Close"].rolling(window=window).mean()

        return df
