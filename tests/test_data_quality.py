import logging

import numpy as np
import pandas as pd

from lib import data_loader as dl
from lib.data_quality import clean_price_data, find_price_errors
from tests.test_data_loader import FakeSource, make_prices


def prices_from_close(close: list[float]) -> pd.DataFrame:
    """終値のリストから、営業日インデックスの OHLC を作る"""
    index = pd.bdate_range("2024-01-01", periods=len(close))
    c = np.array(close, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1000.0, "RawOpen": c},
        index=index,
    )


def test_normal_data_is_unchanged():
    """大きな動きのないデータは、そのまま返す"""
    df = prices_from_close([100, 103, 99, 140, 120, 125])
    found = find_price_errors(df)
    assert found.dropped == [] and found.jumps == []
    pd.testing.assert_frame_equal(clean_price_data(df), df)


def test_reverting_spike_is_dropped(caplog):
    """1306.T の型: 2日だけ 1/10 になって元に戻る行は、取り除いて警告する"""
    df = prices_from_close([376, 377, 36, 36.5, 378, 380])
    with caplog.at_level(logging.WARNING, logger="lib.data_quality"):
        cleaned = clean_price_data(df, "1306.T")
    assert list(cleaned.index) == [df.index[i] for i in [0, 1, 4, 5]]
    assert "1306.T" in caplog.text and f"{df.index[2]:%Y-%m-%d}" in caplog.text


def test_upward_reverting_spike_is_dropped():
    """2倍以上に跳ねて戻る行も、同じく取り除く"""
    df = prices_from_close([100, 101, 1000, 102])
    assert find_price_errors(df).reverting_spikes == [df.index[2]]


def test_non_reverting_jump_is_kept_with_warning(caplog):
    """戻らない急変は、本当の値動きかもしれないので、警告だけで残す"""
    df = prices_from_close([100, 101, 10, 10.5, 11, 10.8, 11.2, 11.5, 11])
    with caplog.at_level(logging.WARNING, logger="lib.data_quality"):
        cleaned = clean_price_data(df, "X")
    pd.testing.assert_frame_equal(cleaned, df)
    assert find_price_errors(df).jumps == [df.index[2]]
    assert "元に戻らない" in caplog.text


def test_spike_longer_than_limit_is_kept():
    """MAX_SPIKE_ROWS 行を超えて続く変化は、戻る急変として扱わない"""
    df = prices_from_close([100, 10, 10, 10, 10, 10, 10, 100])
    found = find_price_errors(df, max_rows=5)
    assert found.reverting_spikes == []
    assert found.jumps == [df.index[1], df.index[7]]


def test_spike_on_last_row_is_only_warned():
    """最後の行の急変は、戻るか分からないので残す"""
    df = prices_from_close([100, 101, 10])
    found = find_price_errors(df)
    assert found.dropped == [] and found.jumps == [df.index[2]]


def test_non_positive_close_is_dropped(caplog):
    """8919.T の型: 終値が 0 以下の行は取り除く。比べる相手は直前の正の終値"""
    df = prices_from_close([100, -5, 0, 101, 102])
    with caplog.at_level(logging.WARNING, logger="lib.data_quality"):
        cleaned = clean_price_data(df, "8919.T")
    assert list(cleaned.index) == [df.index[i] for i in [0, 3, 4]]
    assert "0 以下" in caplog.text
    assert find_price_errors(df).jumps == []


def test_nan_close_is_ignored():
    """終値が NaN の行は検査せず、そのまま残す"""
    df = prices_from_close([100, np.nan, 101])
    pd.testing.assert_frame_equal(clean_price_data(df), df)


def test_fetch_stock_data_drops_errors_in_requested_period(tmp_path, monkeypatch):
    """fetch_stock_data は期間で絞ってから検査し、キャッシュには元のデータを残す"""
    monkeypatch.setattr(dl, "CACHE_DIR", str(tmp_path))
    full = make_prices("2024-01-01", 30)
    bad_day = full.index[20]
    full.loc[bad_day, ["Open", "High", "Low", "Close", "RawOpen"]] /= 10
    source = FakeSource(full)

    df = dl.fetch_stock_data("T", source=source, mode="refresh")
    assert bad_day not in df.index and len(df) == 29
    cached = pd.read_parquet(tmp_path / "fake" / "T.parquet")
    assert bad_day in cached.index

    unchecked = dl.fetch_stock_data("T", source=source, mode="offline", check_quality=False)
    assert bad_day in unchecked.index
