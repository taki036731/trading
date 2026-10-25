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


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", str(tmp_path))
    return tmp_path


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
    cached.iloc[-1, cached.columns.get_loc("Close")] += 50
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
        df["RawOpen"].values, [1000.0, 1010.0, 204.0, 206.0, 208.0]
    )
    ratio = raw["Adj Close"] / raw["Close"]
    np.testing.assert_allclose(df["Open"].values, (raw["Open"] * ratio).values)
    np.testing.assert_allclose(df["Close"].values, raw["Adj Close"].values)
