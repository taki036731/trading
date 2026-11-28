import json
import logging
import os
from datetime import datetime, timedelta
from typing import Literal, cast, get_args

import numpy as np
import pandas as pd

from lib.data_sources import AbstractPriceSource, YFinanceSource

CACHE_DIR = "data"

# 差分取得のとき、キャッシュの最終日から何日さかのぼって取り直すか（暦日）
OVERLAP_DAYS = 7
# 重なった行の価格を比較するときの許容誤差（相対誤差）
PRICE_RTOL = 1e-5
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# 取得モード（#66）
# - auto: 最後に取得してから max_age_days 日以内なら取得元に問い合わせない。取得に失敗したらキャッシュを使う
# - offline: 取得元に一切問い合わせない（キャッシュだけを使う）
# - refresh: キャッシュを使わず全期間を取り直す
DataMode = Literal["auto", "offline", "refresh"]
DATA_MODES: tuple[str, ...] = get_args(DataMode)
DEFAULT_DATA_MODE: DataMode = "auto"
# 引数で指定しないときの取得モード・キャッシュの有効期限を、環境変数で切り替える（ツールのコードを変えずに済むように）
DATA_MODE_ENV = "TRADING_DATA_MODE"
MAX_AGE_ENV = "TRADING_CACHE_MAX_AGE_DAYS"
# auto のとき、最後に取得してから何日以内なら取得元に問い合わせないか（作業は週1回程度のため）
DEFAULT_MAX_AGE_DAYS = 7.0
# 最終取得時刻のメタ情報のファイル名の接尾辞（`<ticker>.meta.json`）
META_SUFFIX = ".meta.json"

logger = logging.getLogger(__name__)


def resolve_data_mode(mode: str | None = None, refresh: bool = False) -> DataMode:
    """
    取得モードを決めます。refresh=True、引数 mode、環境変数 TRADING_DATA_MODE、既定値（auto）の順に優先します。

    Args:
        mode (str, optional): 取得モード（'auto' / 'offline' / 'refresh'）。
        refresh (bool, optional): True なら mode に関係なく 'refresh'（以前からの引数との互換のため）。

    Returns:
        DataMode: 取得モード。

    Raises:
        ValueError: 取得モードの値が不正な場合。
    """
    if refresh:
        return "refresh"
    if mode is None:
        mode = os.environ.get(DATA_MODE_ENV) or DEFAULT_DATA_MODE
    mode = mode.strip().lower()
    if mode not in DATA_MODES:
        raise ValueError(
            f"取得モードが不正です: {mode!r}（{' / '.join(DATA_MODES)} のいずれか。"
            f"環境変数 {DATA_MODE_ENV} も確認してください）"
        )
    return cast(DataMode, mode)


def resolve_max_age_days(max_age_days: float | None = None) -> float:
    """
    キャッシュの有効期限（日数）を決めます。引数、環境変数 TRADING_CACHE_MAX_AGE_DAYS、既定値の順に優先します。

    Args:
        max_age_days (float, optional): 有効期限（日数）。

    Returns:
        float: 有効期限（日数）。

    Raises:
        ValueError: 環境変数の値が数値でない、または負の場合。
    """
    if max_age_days is None:
        env = os.environ.get(MAX_AGE_ENV)
        try:
            max_age_days = float(env) if env else DEFAULT_MAX_AGE_DAYS
        except ValueError:
            raise ValueError(f"環境変数 {MAX_AGE_ENV} が数値ではありません: {env!r}") from None
    if max_age_days < 0:
        raise ValueError(f"キャッシュの有効期限は0以上にしてください: {max_age_days}")
    return max_age_days


def fetch_stock_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    source: AbstractPriceSource | None = None,
    mode: DataMode | None = None,
    max_age_days: float | None = None,
) -> pd.DataFrame:
    """指定された銘柄のヒストリカルデータを取得し、Parquet形式でキャッシュ・更新を行う関数。

    ローカルの `data/<取得元の名前>` ディレクトリにParquetファイルが存在する場合はそれを読み込みます。
    取得モード（mode）が auto（既定）のときは、最後に取得元へ問い合わせてから max_age_days 日を過ぎていれば、
    不足している新しいデータを取得元から取得してキャッシュに追記（差分更新）します。
    期限内なら取得元には問い合わせません。取得に失敗した場合は、キャッシュがあればそれを使います。
    offline のときは取得元に一切問い合わせず、refresh のときはキャッシュを使わず全期間を取り直します。

    調整後価格（auto_adjust=True）は、新しい株式分割・配当があると過去全体が遡って調整し直されます。
    そのため差分取得ではキャッシュの最終日の数日前から取り直し、重なった行の価格がキャッシュと
    ずれていれば、全期間を取り直してキャッシュを置き換えます。

    最後に取得元へ問い合わせた時刻は、`<ticker>.meta.json` に記録します（メタ情報のないキャッシュは期限切れ扱い）。

    Args:
        ticker (str): 取得する銘柄のティッカーシンボル（例: '7203.T', 'AAPL'）。
        start (str, optional): データの取得開始日（'YYYY-MM-DD'）。デフォルトは None（全期間）。
        end (str, optional): データの取得終了日（'YYYY-MM-DD'）。デフォルトは None（最新まで）。
        refresh (bool, optional): True の場合、mode に関係なく全期間を取り直します（mode='refresh' と同じ）。
            デフォルトは False。
        source (AbstractPriceSource, optional): 株価データの取得元。デフォルトは None（YFinanceSource）。
        mode (DataMode, optional): 取得モード（'auto' / 'offline' / 'refresh'）。
            デフォルトは None（環境変数 TRADING_DATA_MODE、なければ 'auto'）。
        max_age_days (float, optional): auto のときのキャッシュの有効期限（日数）。
            デフォルトは None（環境変数 TRADING_CACHE_MAX_AGE_DAYS、なければ7日）。

    Returns:
        pd.DataFrame: 指定された期間の株価データを含むデータフレーム。
                      データの取得に失敗した場合や存在しない場合は空のデータフレームを返します。

    Raises:
        FileNotFoundError: offline のときに、使えるキャッシュがない場合。
        Exception: キャッシュがなく、取得元からの取得が例外になった場合（その例外）。
    """
    if source is None:
        source = YFinanceSource()
    data_mode = resolve_data_mode(mode, refresh)
    max_age = resolve_max_age_days(max_age_days)

    # 取得元ごとに価格の調整方法が違う可能性があるため、キャッシュは取得元ごとに分ける
    cache_dir = os.path.join(CACHE_DIR, source.name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}.parquet")
    meta_path = os.path.join(cache_dir, f"{ticker}{META_SUFFIX}")

    # 1. キャッシュの読み込み
    df = None
    if os.path.exists(cache_path) and data_mode != "refresh":
        logger.info(f"[{ticker}] キャッシュからデータを読み込みます。")
        df = pd.read_parquet(cache_path, engine="auto")

    if df is not None and not set(source.required_columns) <= set(df.columns):
        logger.warning(
            f"[{ticker}] キャッシュに必須カラムが不足しています。全期間を取り直します。"
        )
        df = None

    # 2. 取得元への問い合わせ（差分更新、または全期間の新規取得）
    fetched_at = _read_fetched_at(meta_path)
    if data_mode == "offline":
        if df is None:
            raise FileNotFoundError(
                f"[{ticker}] オフラインモードですが、使えるキャッシュがありません: {cache_path}"
            )
    elif df is not None and data_mode == "auto" and _is_fresh(fetched_at, max_age):
        logger.info(
            f"[{ticker}] 最後の取得（{fetched_at:%Y-%m-%d %H:%M}）から {max_age:g} 日以内のため、"
            "取得元には問い合わせません。"
        )
    else:
        try:
            df = _update_from_source(ticker, df, source, cache_path)
        except Exception as e:  # noqa: BLE001
            if df is None:
                raise
            # 通信エラーや利用制限のときは、古くてもキャッシュで続ける（auto のみ。refresh では df は None）
            logger.warning(
                f"[{ticker}] 取得元からの取得に失敗したため、キャッシュを使います。：{e!r}"
            )
        else:
            if df is None:
                return pd.DataFrame()
            fetched_at = _now()
            _write_fetched_at(meta_path, fetched_at)

    # オフラインではキャッシュの時点で結果が変わるため、データの最終日を残す（#66）
    fetched_note = f"、最終取得: {fetched_at:%Y-%m-%d %H:%M}" if fetched_at else ""
    logger.info(f"[{ticker}] データの最終日: {df.index.max():%Y-%m-%d}{fetched_note}")

    # 3. 指定された期間でデータを絞り込んで返す
    if start and end:
        df = df.loc[start:end]
    elif start:
        df = df.loc[start:]
    elif end:
        df = df.loc[:end]

    return df


def _update_from_source(
    ticker: str,
    cached: pd.DataFrame | None,
    source: AbstractPriceSource,
    cache_path: str,
) -> pd.DataFrame | None:
    """取得元からデータを取得し、キャッシュを更新する内部関数。

    キャッシュがあれば差分更新し、調整のずれがあるかキャッシュがなければ全期間を取得します。

    Args:
        ticker (str): 銘柄のティッカーシンボル。
        cached (pd.DataFrame, optional): キャッシュのデータ。None なら全期間を取得します。
        source (AbstractPriceSource): 株価データの取得元。
        cache_path (str): キャッシュの Parquet ファイルのパス。

    Returns:
        pd.DataFrame | None: 更新後のデータ。全期間の取得でデータがなかった場合は None。
    """
    df = cached
    if df is not None:
        # キャッシュの最終日の数日前から取り直し、調整のずれを確認できるようにする
        last_date = df.index.max()
        fetch_start = (last_date - pd.Timedelta(days=OVERLAP_DAYS)).strftime(
            "%Y-%m-%d"
        )
        df_new = source.fetch(ticker, fetch_start)

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
        logger.info(f"[{ticker}] {source.name} から全期間取得します。")
        # キャッシュ構築のため、全期間を取得
        df = source.fetch(ticker)

        if df.empty:
            logger.warning(
                f"[{ticker}] データの取得に失敗したか、データが存在しません。"
            )
            return None
        df.to_parquet(cache_path, engine="auto")
        logger.info(f"[{ticker}] 全期間データをキャッシュに保存しました。")

    return df


def _now() -> datetime:
    """現在時刻を返す内部関数（テストで差し替えるため）。"""
    return datetime.now()


def _is_fresh(fetched_at: datetime | None, max_age_days: float) -> bool:
    """最終取得時刻が有効期限内か判定する内部関数。メタ情報がなければ期限切れとします。"""
    if fetched_at is None:
        return False
    return _now() - fetched_at < timedelta(days=max_age_days)


def _read_fetched_at(meta_path: str) -> datetime | None:
    """メタ情報から最終取得時刻を読む内部関数。ファイルがない・読めない場合は None。"""
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return datetime.fromisoformat(json.load(f)["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"メタ情報を読めないため、期限切れとして扱います: {meta_path}：{e!r}")
        return None


def _write_fetched_at(meta_path: str, fetched_at: datetime) -> None:
    """最終取得時刻をメタ情報に書く内部関数。"""
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": fetched_at.isoformat(timespec="seconds")}, f)


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

