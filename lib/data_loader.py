import logging
import os

import pandas as pd
import yfinance as yf

CACHE_DIR = "data"

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """指定された銘柄のヒストリカルデータを取得し、Parquet形式でキャッシュ・更新を行う関数。

    ローカルの `data` ディレクトリにParquetファイルが存在する場合はそれを読み込みます。
    キャッシュが存在する場合でも、データ内の最新日付を確認し、不足している新しいデータがあれば
    yfinanceから自動的に取得してキャッシュに追記（差分更新）します。

    Args:
        ticker (str): 取得する銘柄のティッカーシンボル（例: '7203.T', 'AAPL'）。
        start (str, optional): データの取得開始日（'YYYY-MM-DD'）。デフォルトは None（全期間）。
        end (str, optional): データの取得終了日（'YYYY-MM-DD'）。デフォルトは None（最新まで）。

    Returns:
        pd.DataFrame: 指定された期間の株価データを含むデータフレーム。
                      データの取得に失敗した場合や存在しない場合は空のデータフレームを返します。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")

    # 1. データの用意（キャッシュの読み込みと差分更新、または全期間の新規取得）
    if os.path.exists(cache_path):
        logger.info(f"[{ticker}] キャッシュからデータを読み込みます。")
        df = pd.read_parquet(cache_path, engine="auto")

        # キャッシュの最新日付を取得し、その翌日を差分取得の開始日に設定
        last_date = df.index.max()
        next_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 次の日付から最新までのデータを取得
        df_new = _history(ticker, next_date)

        if not df_new.empty:
            logger.info(
                f"[{ticker}] {next_date} 以降の新しいデータを取得し、キャッシュを更新します。"
            )
            # 既存データと新規データを結合
            df = pd.concat([df, df_new])
            # 重複データが存在する場合は排除（念のための安全策）し、日付順にソート
            df = df[~df.index.duplicated(keep="last")].sort_index()
            # 更新したデータセットで上書き保存
            df.to_parquet(cache_path, engine="auto")
        else:
            logger.info(
                f"[{ticker}] 追加すべき新しいデータはありません（キャッシュは最新です）。"
            )

    else:
        logger.info(
            f"[{ticker}] キャッシュが存在しないため、yfinanceから全期間取得します。"
        )
        # キャッシュ構築のため、最初は全期間 (period="max") を取得
        df = _history(ticker)

        if not df.empty:
            df.to_parquet(cache_path, engine="auto")
            logger.info(f"[{ticker}] 全期間データをキャッシュに保存しました。")
        else:
            logger.warning(
                f"[{ticker}] データの取得に失敗したか、データが存在しません。"
            )
            return df

    # 2. 指定された期間でデータを絞り込んで返す
    if start and end:
        df = df.loc[start:end]
    elif start:
        df = df.loc[start:]
    elif end:
        df = df.loc[:end]

    return df


def _history(ticker: str, start: str | None = None) -> pd.DataFrame:
    """yfinanceを使用してヒストリカルデータを取得する内部関数。
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
