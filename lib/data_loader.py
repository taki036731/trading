import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = "data"

# 差分取得のとき、キャッシュの最終日から何日さかのぼって取り直すか（暦日）
OVERLAP_DAYS = 7
# 重なった行の価格を比較するときの許容誤差（相対誤差）
PRICE_RTOL = 1e-5
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """指定された銘柄のヒストリカルデータを取得し、Parquet形式でキャッシュ・更新を行う関数。

    ローカルの `data` ディレクトリにParquetファイルが存在する場合はそれを読み込みます。
    キャッシュが存在する場合でも、データ内の最新日付を確認し、不足している新しいデータがあれば
    yfinanceから自動的に取得してキャッシュに追記（差分更新）します。

    調整後価格（auto_adjust=True）は、新しい株式分割・配当があると過去全体が遡って調整し直されます。
    そのため差分取得ではキャッシュの最終日の数日前から取り直し、重なった行の価格がキャッシュと
    ずれていれば、全期間を取り直してキャッシュを置き換えます。

    Args:
        ticker (str): 取得する銘柄のティッカーシンボル（例: '7203.T', 'AAPL'）。
        start (str, optional): データの取得開始日（'YYYY-MM-DD'）。デフォルトは None（全期間）。
        end (str, optional): データの取得終了日（'YYYY-MM-DD'）。デフォルトは None（最新まで）。
        refresh (bool, optional): True の場合、キャッシュを使わず全期間を取り直します。デフォルトは False。

    Returns:
        pd.DataFrame: 指定された期間の株価データを含むデータフレーム。
                      データの取得に失敗した場合や存在しない場合は空のデータフレームを返します。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")

    # 1. データの用意（キャッシュの読み込みと差分更新、または全期間の新規取得）
    df = None
    if os.path.exists(cache_path) and not refresh:
        logger.info(f"[{ticker}] キャッシュからデータを読み込みます。")
        df = pd.read_parquet(cache_path, engine="auto")

        # キャッシュの最終日の数日前から取り直し、調整のずれを確認できるようにする
        last_date = df.index.max()
        fetch_start = (last_date - pd.Timedelta(days=OVERLAP_DAYS)).strftime(
            "%Y-%m-%d"
        )
        df_new = _history(ticker, fetch_start)

        if df_new.empty:
            logger.info(
                f"[{ticker}] 追加すべき新しいデータはありません（キャッシュは最新です）。"
            )
        elif _is_adjustment_changed(df, df_new):
            logger.warning(
                f"[{ticker}] キャッシュの価格と取得した価格がずれています"
                "（株式分割・配当による再調整の可能性）。全期間を取り直します。"
            )
            df = None
        else:
            logger.info(
                f"[{ticker}] {fetch_start} 以降のデータを取得し、キャッシュを更新します。"
            )
            # 既存データと新規データを結合（重なった日は新しい値を優先）し、日付順にソート
            df = pd.concat([df, df_new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            # 更新したデータセットで上書き保存
            df.to_parquet(cache_path, engine="auto")

    if df is None:
        logger.info(f"[{ticker}] yfinanceから全期間取得します。")
        # キャッシュ構築のため、全期間 (period="max") を取得
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


def _is_adjustment_changed(cached: pd.DataFrame, fetched: pd.DataFrame) -> bool:
    """キャッシュと新しく取得したデータの重なった行を比べ、調整後価格がずれているか判定する内部関数。

    キャッシュの最終行は、場中に取得した確定前の値の可能性があるため比較から除きます。

    Args:
        cached (pd.DataFrame): キャッシュから読み込んだデータ。
        fetched (pd.DataFrame): 差分取得したデータ（キャッシュの最終日より前の日を含む）。

    Returns:
        bool: 重なった行の Open/High/Low/Close のいずれかが許容誤差を超えてずれていれば True。
    """
    common = cached.index[:-1].intersection(fetched.index)
    if common.empty:
        return False

    old = cached.loc[common, PRICE_COLUMNS].to_numpy(dtype=float)
    new = fetched.loc[common, PRICE_COLUMNS].to_numpy(dtype=float)
    return not np.allclose(old, new, rtol=PRICE_RTOL, atol=0.0, equal_nan=True)


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
