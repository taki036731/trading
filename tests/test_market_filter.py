"""
地合いフィルタ（ExternalIndicator + ColumnAboveSignal）のテスト。
"""

import logging

import numpy as np
import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import (
    AbstractIndicator,
    ATRIndicator,
    ExternalIndicator,
    MAIndicator,
)
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import AbstractSignal, ColumnAboveSignal, GoldenCrossSignal

from .helpers import create_random_walk_df

MARKET_COLS = ["mkt_sma50", "mkt_sma200"]


def _market_ma() -> MAIndicator:
    return MAIndicator("SMA", 50, 200, short_col="mkt_sma50", long_col="mkt_sma200")


def _market_filter(source: pd.DataFrame) -> ExternalIndicator:
    return ExternalIndicator(source, _market_ma(), name="TOPIX")


def _source_sma(source: pd.DataFrame) -> pd.DataFrame:
    """参照元で直接計算した SMA（期待値）。"""
    df = source.copy()
    _market_ma().generate(df)
    return df[MARKET_COLS]


def test_external_indicator_adds_source_values_by_date():
    """
    参照元で計算した値が、売買する銘柄の同じ日付の行に入ることをテスト。
    参照元の全期間で計算するので、銘柄のデータが200行より短くても200日線が入る
    """
    source = create_random_walk_df(days=400, seed=1)
    df = create_random_walk_df(days=400, seed=2).iloc[300:].copy()

    _market_filter(source).generate(df)

    expected = _source_sma(source).loc[df.index]
    pd.testing.assert_frame_equal(df[MARKET_COLS], expected)
    assert df["mkt_sma200"].notna().all()


def test_external_indicator_does_not_modify_source():
    """
    参照元のデータフレームに列が追加されないことと、繰り返し使えることをテスト
    """
    source = create_random_walk_df(days=300, seed=1)
    original_columns = list(source.columns)
    indicator = _market_filter(source)

    for _ in range(2):
        indicator.generate(create_random_walk_df(days=300, seed=2))

    assert list(source.columns) == original_columns


def test_external_indicator_fills_missing_source_days_with_previous_value():
    """
    参照元に行がない日は、前の日の値（過去方向のみ）で埋めることをテスト
    """
    full = create_random_walk_df(days=300, seed=1)
    missing_day = full.index[250]
    source = full.drop(index=missing_day)
    df = create_random_walk_df(days=300, seed=2)

    _market_filter(source).generate(df)

    expected = _source_sma(source)
    previous_day = full.index[249]
    np.testing.assert_allclose(
        df.loc[[missing_day], MARKET_COLS].to_numpy(),
        expected.loc[[previous_day], MARKET_COLS].to_numpy(),
    )


def test_external_indicator_is_nan_outside_source_period(caplog):
    """
    参照元の最初の日より前と、最終日より後の日は NaN になり、後者は警告が出ることをテスト
    """
    full = create_random_walk_df(days=400, seed=1)
    source = full.iloc[50:350]
    df = create_random_walk_df(days=400, seed=2)

    with caplog.at_level(logging.WARNING):
        _market_filter(source).generate(df)

    assert df[MARKET_COLS].iloc[:50].isna().all().all()
    assert df[MARKET_COLS].iloc[350:].isna().all().all()
    assert df["mkt_sma200"].iloc[249:350].notna().all()
    assert "参照元のデータが" in caplog.text
    assert "[TOPIX]" in caplog.text


def test_external_indicator_rejects_existing_column():
    """
    出力列が既にある場合は、黙って上書きせずに ValueError になることをテスト
    """
    source = create_random_walk_df(days=300, seed=1)
    df = create_random_walk_df(days=300, seed=2)
    df["mkt_sma50"] = 1.0

    with pytest.raises(ValueError, match="mkt_sma50"):
        _market_filter(source).generate(df)


def test_external_indicator_validates_source_columns():
    """
    参照元に必要なカラム（Close）がない場合は ValueError になることをテスト
    """
    source = create_random_walk_df(days=300, seed=1).drop(columns=["Close"])
    df = create_random_walk_df(days=300, seed=2)

    with pytest.raises(ValueError, match="Close"):
        _market_filter(source).generate(df)


def test_column_above_signal_is_true_while_above_and_shifted():
    """
    left > right の間ずっと True になり（クロスの日だけではない）、1日シフトされることと、
    NaN の日は False になることをテスト
    """
    index = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame(
        {
            "a": [np.nan, 2.0, 2.0, 0.5, 3.0, 3.0],
            "b": [1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
        },
        index=index,
    )

    signal = ColumnAboveSignal("a", "b").generate(df)

    # _calculate: [F, T, T, F, T, F] を1日シフト
    assert signal.tolist() == [False, False, True, True, False, True]


def _backtester(source: pd.DataFrame | None) -> Backtester:
    indicators: list[AbstractIndicator] = [MAIndicator("EMA", 5, 20), ATRIndicator(15)]
    entries: list[AbstractSignal] = [GoldenCrossSignal()]
    if source is not None:
        indicators.append(_market_filter(source))
        entries.append(ColumnAboveSignal("mkt_sma50", "mkt_sma200"))
    return Backtester(
        indicators=indicators,
        entries=entries,
        exits=[],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


def test_market_filter_only_removes_entries_on_bearish_days():
    """
    地合いフィルタを足すと、前日の参照元の 50日線 > 200日線 でない日のエントリーだけが消えることをテスト
    """
    source = create_random_walk_df(days=600, seed=3)
    df = create_random_walk_df(days=600, seed=4)

    without = _backtester(None).prepare(df).entries
    with_filter = _backtester(source).prepare(df).entries

    sma = _source_sma(source)
    bullish_prev_day = (sma["mkt_sma50"] > sma["mkt_sma200"]).shift(1, fill_value=False)
    pd.testing.assert_series_equal(with_filter, without & bullish_prev_day, check_names=False)
    # テストが意味を持つように、消えたエントリーと残ったエントリーの両方がある
    assert (without & ~with_filter).any()
    assert with_filter.any()


@pytest.mark.parametrize("k", [250, 300, 400])
def test_market_filter_does_not_use_future_source_prices(k):
    """
    参照元の未来の価格（k日目の終値以降）を書き換えても、k日目までのエントリーが変わらないことをテスト
    """
    source = create_random_walk_df(days=600, seed=3)
    df = create_random_walk_df(days=600, seed=4)
    changed = source.copy()
    changed.loc[changed.index[k], ["High", "Low", "Close"]] *= 1.3
    changed.iloc[k + 1 :] *= 0.7

    original = _backtester(source).prepare(df).entries
    after = _backtester(changed).prepare(df).entries

    pd.testing.assert_series_equal(original.iloc[: k + 1], after.iloc[: k + 1])
    assert not original.iloc[k + 1 :].equals(after.iloc[k + 1 :])
