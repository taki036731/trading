from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from lib import data_loader as dl
from lib.data_sources import AbstractPriceSource
from lib.data_sources.yfinance_source import to_adjusted_with_raw


def make_prices(start: str, periods: int, base: float = 1000.0) -> pd.DataFrame:
    """営業日インデックスの合成 OHLCV を作る"""
    index = pd.bdate_range(start, periods=periods)
    close = base + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(periods, 1000.0),
            "RawOpen": close - 1,
        },
        index=index,
    )


class FakeSource(AbstractPriceSource):
    """取得元の差し替え。取得元が返す「現在の」全期間データを持ち、呼び出しを記録する"""

    def __init__(self, full: pd.DataFrame):
        self.full = full
        self.calls: list[str | None] = []

    @property
    def name(self) -> str:
        return "fake"

    def _fetch(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        self.calls.append(start)
        if start is None:
            return self.full.copy()
        return self.full.loc[start:].copy()


@pytest.fixture(autouse=True)
def clear_mode_env(monkeypatch):
    """手元で環境変数を設定していても、テストの結果が変わらないようにする"""
    monkeypatch.delenv(dl.DATA_MODE_ENV, raising=False)
    monkeypatch.delenv(dl.MAX_AGE_ENV, raising=False)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", str(tmp_path))
    return tmp_path


class FailingSource(FakeSource):
    """取得元への問い合わせが必ず失敗する（通信エラーや利用制限の代わり）"""

    def _fetch(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        self.calls.append(start)
        raise ConnectionError("network")


def set_now(monkeypatch, now: str) -> None:
    monkeypatch.setattr(dl, "_now", lambda: datetime.fromisoformat(now))


def meta_path(cache_dir, ticker: str):
    return cache_dir / "fake" / f"{ticker}{dl.META_SUFFIX}"


def write_cache(cache_dir, ticker: str, df: pd.DataFrame) -> None:
    (cache_dir / "fake").mkdir(exist_ok=True)
    df.to_parquet(cache_path(cache_dir, ticker))


def cache_path(cache_dir, ticker: str):
    return cache_dir / "fake" / f"{ticker}.parquet"


def test_appends_only_new_rows_when_not_adjusted(cache_dir):
    """調整がなければ差分だけが追記され、全期間の取得は呼ばれない"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20])
    fake = FakeSource(full)

    df = dl.fetch_stock_data("TEST", source=fake)

    assert None not in fake.calls
    pd.testing.assert_frame_equal(df, full, check_freq=False)
    cached = pd.read_parquet(cache_path(cache_dir, "TEST"))
    pd.testing.assert_frame_equal(cached, full, check_freq=False)


def test_refetches_full_history_after_split(cache_dir):
    """キャッシュ作成後に 1:5 分割があれば全期間を取り直し、分割日に偽の暴落が入らない"""
    before_split = make_prices("2024-01-01", 30, base=5000.0)
    write_cache(cache_dir, "TEST", before_split.iloc[:20])

    # 分割後の yfinance は、過去の価格も 1/5 に調整し直して返す
    adjusted = before_split.copy()
    adjusted[dl.PRICE_COLUMNS] = adjusted[dl.PRICE_COLUMNS] / 5
    fake = FakeSource(adjusted)

    df = dl.fetch_stock_data("TEST", source=fake)

    assert fake.calls[-1] is None
    pd.testing.assert_frame_equal(df, adjusted, check_freq=False)
    assert df["Close"].pct_change().min() > -0.5
    cached = pd.read_parquet(cache_path(cache_dir, "TEST"))
    pd.testing.assert_frame_equal(cached, adjusted, check_freq=False)


def test_last_cached_row_is_replaced_without_full_refetch(cache_dir):
    """キャッシュの最終行だけ値が違う（場中の確定前の値）場合は全期間を取り直さず、新しい値で置き換える"""
    full = make_prices("2024-01-01", 30)
    cached = full.iloc[:20].copy()
    cached.loc[cached.index[-1:], "Close"] += 50
    write_cache(cache_dir, "TEST", cached)
    fake = FakeSource(full)

    df = dl.fetch_stock_data("TEST", source=fake)

    assert None not in fake.calls
    pd.testing.assert_frame_equal(df, full, check_freq=False)


def test_refresh_always_fetches_full_history(cache_dir):
    """refresh=True ならキャッシュがあっても全期間を取得する"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20])
    fake = FakeSource(full)

    df = dl.fetch_stock_data("TEST", refresh=True, source=fake)

    assert fake.calls == [None]
    pd.testing.assert_frame_equal(df, full, check_freq=False)


def test_is_adjustment_changed_tolerates_tiny_float_noise():
    """浮動小数点の誤差程度のずれは、再調整とみなさない"""
    cached = make_prices("2024-01-01", 10)
    fetched = cached.copy()
    fetched[dl.PRICE_COLUMNS] = fetched[dl.PRICE_COLUMNS] * (1 + 1e-9)

    assert not dl._is_adjustment_changed(cached, fetched)


def test_is_adjustment_changed_detects_dividend_adjustment():
    """配当による 1% 程度の再調整は検出する"""
    cached = make_prices("2024-01-01", 10)
    fetched = cached.copy()
    fetched[dl.PRICE_COLUMNS] = fetched[dl.PRICE_COLUMNS] * 0.99

    assert dl._is_adjustment_changed(cached, fetched)


def test_fetch_raises_when_required_column_missing():
    """取得元が必須カラムを返さなければエラーにする"""
    fake = FakeSource(make_prices("2024-01-01", 10).drop(columns=["Volume"]))

    with pytest.raises(ValueError, match="Volume"):
        fake.fetch("TEST")


def test_fetch_returns_empty_frame_without_validation():
    """データが存在しない場合は、検証せずに空のデータフレームを返す"""
    fake = FakeSource(pd.DataFrame())

    assert fake.fetch("TEST").empty


def test_cache_is_separated_by_source(cache_dir):
    """キャッシュは取得元の名前のフォルダに保存される"""
    full = make_prices("2024-01-01", 10)

    dl.fetch_stock_data("TEST", source=FakeSource(full))

    assert cache_path(cache_dir, "TEST").exists()
    assert not (cache_dir / "TEST.parquet").exists()


def test_refetches_full_history_when_cache_lacks_required_columns(cache_dir):
    """必須カラム（RawOpen など）のない古いキャッシュは、全期間を取り直して置き換える"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20].drop(columns=["RawOpen"]))
    fake = FakeSource(full)

    df = dl.fetch_stock_data("TEST", source=fake)

    assert fake.calls == [None]
    pd.testing.assert_frame_equal(df, full, check_freq=False)


def test_to_adjusted_with_raw_restores_pre_split_price():
    """
    yfinance の未調整データ（分割調整のみ）から、調整後の OHLC と実際の始値を作る。
    3日目に 1:5 分割があれば、それより前の実際の始値は yfinance の値の5倍になる
    """
    index = pd.bdate_range("2024-01-01", periods=5)
    raw = pd.DataFrame(
        {
            "Open": [200.0, 202.0, 204.0, 206.0, 208.0],
            "High": [210.0, 212.0, 214.0, 216.0, 218.0],
            "Low": [190.0, 192.0, 194.0, 196.0, 198.0],
            "Close": [205.0, 207.0, 209.0, 211.0, 213.0],
            # 配当の調整で、過去ほど調整後終値が小さい
            "Adj Close": [205.0 * 0.9, 207.0 * 0.9, 209.0 * 0.95, 211.0, 213.0],
            "Volume": [1000.0] * 5,
            "Dividends": [0.0] * 5,
            "Stock Splits": [0.0, 0.0, 5.0, 0.0, 0.0],
        },
        index=index,
    )

    df = to_adjusted_with_raw(raw)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "RawOpen"]
    np.testing.assert_allclose(
        df["RawOpen"].to_numpy(), [1000.0, 1010.0, 204.0, 206.0, 208.0]
    )
    ratio = raw["Adj Close"] / raw["Close"]
    np.testing.assert_allclose(df["Open"].to_numpy(), (raw["Open"] * ratio).to_numpy())
    np.testing.assert_allclose(df["Close"].to_numpy(), raw["Adj Close"].to_numpy())


def test_auto_skips_source_within_max_age(cache_dir, monkeypatch):
    """auto では、最後の取得から有効期限内なら取得元に問い合わせず、キャッシュをそのまま返す"""
    full = make_prices("2024-01-01", 30)
    fake = FakeSource(full.iloc[:20])
    set_now(monkeypatch, "2024-02-01T10:00:00")
    dl.fetch_stock_data("TEST", source=fake)
    assert fake.calls == [None]
    assert meta_path(cache_dir, "TEST").exists()

    # 取得元のデータが増えても、7日以内なら問い合わせない
    fake.full = full
    set_now(monkeypatch, "2024-02-08T09:59:00")
    df = dl.fetch_stock_data("TEST", source=fake)

    assert fake.calls == [None]
    pd.testing.assert_frame_equal(df, full.iloc[:20], check_freq=False)


def test_auto_updates_after_max_age(cache_dir, monkeypatch):
    """auto では、有効期限を過ぎていれば差分取得して、最終取得時刻を更新する"""
    full = make_prices("2024-01-01", 30)
    fake = FakeSource(full.iloc[:20])
    set_now(monkeypatch, "2024-02-01T10:00:00")
    dl.fetch_stock_data("TEST", source=fake)

    fake.full = full
    set_now(monkeypatch, "2024-02-08T10:00:00")
    df = dl.fetch_stock_data("TEST", source=fake)

    assert len(fake.calls) == 2 and fake.calls[-1] is not None
    pd.testing.assert_frame_equal(df, full, check_freq=False)
    assert "2024-02-08T10:00:00" in meta_path(cache_dir, "TEST").read_text()


def test_max_age_from_env(cache_dir, monkeypatch):
    """有効期限は環境変数で変えられる（0 なら毎回問い合わせる）"""
    full = make_prices("2024-01-01", 20)
    fake = FakeSource(full)
    dl.fetch_stock_data("TEST", source=fake)
    monkeypatch.setenv(dl.MAX_AGE_ENV, "0")

    dl.fetch_stock_data("TEST", source=fake)

    assert len(fake.calls) == 2


def test_cache_without_meta_is_treated_as_expired(cache_dir):
    """メタ情報のない既存のキャッシュは期限切れとして扱い、差分取得する"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20])
    fake = FakeSource(full)

    dl.fetch_stock_data("TEST", source=fake)

    assert len(fake.calls) == 1 and fake.calls[0] is not None
    assert meta_path(cache_dir, "TEST").exists()


def test_auto_falls_back_to_cache_when_source_fails(cache_dir, caplog):
    """auto で取得元からの取得に失敗したら、古いキャッシュを使って警告する。最終取得時刻は更新しない"""
    full = make_prices("2024-01-01", 20)
    write_cache(cache_dir, "TEST", full)
    fake = FailingSource(full)

    df = dl.fetch_stock_data("TEST", source=fake)

    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(df, full, check_freq=False)
    assert "キャッシュを使います" in caplog.text
    assert not meta_path(cache_dir, "TEST").exists()


def test_auto_raises_when_source_fails_without_cache(cache_dir):
    """キャッシュがなく取得にも失敗したら、例外をそのまま出す"""
    with pytest.raises(ConnectionError):
        dl.fetch_stock_data("TEST", source=FailingSource(make_prices("2024-01-01", 5)))


def test_offline_never_calls_source(cache_dir):
    """offline では、メタ情報がなく期限切れでも取得元に問い合わせず、キャッシュを返す"""
    full = make_prices("2024-01-01", 20)
    write_cache(cache_dir, "TEST", full)
    fake = FailingSource(full)

    df = dl.fetch_stock_data("TEST", source=fake, mode="offline", start="2024-01-10")

    assert fake.calls == []
    pd.testing.assert_frame_equal(df, full.loc["2024-01-10":], check_freq=False)


def test_offline_from_env_raises_without_cache(cache_dir, monkeypatch):
    """環境変数で offline にでき、キャッシュがなければ FileNotFoundError にする"""
    monkeypatch.setenv(dl.DATA_MODE_ENV, "offline")
    fake = FakeSource(make_prices("2024-01-01", 5))

    with pytest.raises(FileNotFoundError, match="オフライン"):
        dl.fetch_stock_data("TEST", source=fake)
    assert fake.calls == []


def test_mode_refresh_fetches_full_history_even_if_fresh(cache_dir):
    """refresh では、有効期限内でも全期間を取り直す"""
    full = make_prices("2024-01-01", 20)
    fake = FakeSource(full)
    dl.fetch_stock_data("TEST", source=fake)

    dl.fetch_stock_data("TEST", source=fake, mode="refresh")

    assert fake.calls == [None, None]


def test_resolve_data_mode_priority(monkeypatch):
    """refresh=True、引数、環境変数、既定値の順に優先し、不正な値はエラーにする"""
    assert dl.resolve_data_mode() == "auto"
    monkeypatch.setenv(dl.DATA_MODE_ENV, "Offline")
    assert dl.resolve_data_mode() == "offline"
    assert dl.resolve_data_mode("auto") == "auto"
    assert dl.resolve_data_mode("offline", refresh=True) == "refresh"
    monkeypatch.setenv(dl.DATA_MODE_ENV, "online")
    with pytest.raises(ValueError, match="online"):
        dl.resolve_data_mode()


def test_broken_meta_is_treated_as_expired(cache_dir):
    """メタ情報が壊れていても止まらず、期限切れとして扱う"""
    full = make_prices("2024-01-01", 20)
    write_cache(cache_dir, "TEST", full)
    meta_path(cache_dir, "TEST").write_text("not json")
    fake = FakeSource(full)

    dl.fetch_stock_data("TEST", source=fake)

    assert len(fake.calls) == 1
