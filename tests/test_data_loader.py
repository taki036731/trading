import numpy as np
import pandas as pd
import pytest

from lib import data_loader as dl


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
        },
        index=index,
    )


class FakeHistory:
    """`_history` の差し替え。yfinance が返す「現在の」全期間データを持ち、呼び出しを記録する"""

    def __init__(self, full: pd.DataFrame):
        self.full = full
        self.calls: list[str | None] = []

    def __call__(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        self.calls.append(start)
        if start is None:
            return self.full.copy()
        return self.full.loc[start:].copy()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_DIR", str(tmp_path))
    return tmp_path


def write_cache(cache_dir, ticker: str, df: pd.DataFrame) -> None:
    df.to_parquet(cache_dir / f"{ticker}.parquet")


def test_appends_only_new_rows_when_not_adjusted(cache_dir, monkeypatch):
    """調整がなければ差分だけが追記され、全期間の取得は呼ばれない"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20])
    fake = FakeHistory(full)
    monkeypatch.setattr(dl, "_history", fake)

    df = dl.fetch_stock_data("TEST")

    assert None not in fake.calls
    pd.testing.assert_frame_equal(df, full, check_freq=False)
    cached = pd.read_parquet(cache_dir / "TEST.parquet")
    pd.testing.assert_frame_equal(cached, full, check_freq=False)


def test_refetches_full_history_after_split(cache_dir, monkeypatch):
    """キャッシュ作成後に 1:5 分割があれば全期間を取り直し、分割日に偽の暴落が入らない"""
    before_split = make_prices("2024-01-01", 30, base=5000.0)
    write_cache(cache_dir, "TEST", before_split.iloc[:20])

    # 分割後の yfinance は、過去の価格も 1/5 に調整し直して返す
    adjusted = before_split.copy()
    adjusted[dl.PRICE_COLUMNS] = adjusted[dl.PRICE_COLUMNS] / 5
    fake = FakeHistory(adjusted)
    monkeypatch.setattr(dl, "_history", fake)

    df = dl.fetch_stock_data("TEST")

    assert fake.calls[-1] is None
    pd.testing.assert_frame_equal(df, adjusted, check_freq=False)
    assert df["Close"].pct_change().min() > -0.5
    cached = pd.read_parquet(cache_dir / "TEST.parquet")
    pd.testing.assert_frame_equal(cached, adjusted, check_freq=False)


def test_last_cached_row_is_replaced_without_full_refetch(cache_dir, monkeypatch):
    """キャッシュの最終行だけ値が違う（場中の確定前の値）場合は全期間を取り直さず、新しい値で置き換える"""
    full = make_prices("2024-01-01", 30)
    cached = full.iloc[:20].copy()
    cached.iloc[-1, cached.columns.get_loc("Close")] += 50
    write_cache(cache_dir, "TEST", cached)
    fake = FakeHistory(full)
    monkeypatch.setattr(dl, "_history", fake)

    df = dl.fetch_stock_data("TEST")

    assert None not in fake.calls
    pd.testing.assert_frame_equal(df, full, check_freq=False)


def test_refresh_always_fetches_full_history(cache_dir, monkeypatch):
    """refresh=True ならキャッシュがあっても全期間を取得する"""
    full = make_prices("2024-01-01", 30)
    write_cache(cache_dir, "TEST", full.iloc[:20])
    fake = FakeHistory(full)
    monkeypatch.setattr(dl, "_history", fake)

    df = dl.fetch_stock_data("TEST", refresh=True)

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
