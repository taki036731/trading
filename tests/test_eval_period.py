# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false
"""
評価期間（start / end）の指定のテスト（#38）。

評価期間より前のデータはインジケーターのウォームアップにだけ使い、
シミュレーションと年率の指標は評価期間だけで行うことを確かめる。
"""

import logging

import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.metrics import calc_metrics
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
    InDateRangeSignal,
)

from .helpers import create_random_walk_df

START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2024-12-31")
# 取引の比較に使う列（Id 系の列は除く）
TRADE_COLUMNS = [
    "Entry Timestamp",
    "Size",
    "Avg Entry Price",
    "Exit Timestamp",
    "Avg Exit Price",
    "PnL",
    "Status",
]


def build_strategy(
    begin: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> Backtester:
    """strategy01 に近い戦略。begin / end を渡すと、InDateRangeSignal でエントリーの期間を絞る"""
    entries = [GoldenCrossSignal(), ATRConditionSignal(lower=0.5, upper=50)]
    if begin is not None and end is not None:
        entries.append(InDateRangeSignal(begin.to_pydatetime(), end.to_pydatetime()))
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries=entries,
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=10_000_000,
    )


def _trades(portfolio) -> pd.DataFrame:
    return portfolio.trades.records_readable[TRADE_COLUMNS].reset_index(drop=True)


@pytest.fixture
def df() -> pd.DataFrame:
    return create_random_walk_df(1500, seed=3)


def test_same_trades_as_date_range_signal(df: pd.DataFrame):
    """
    評価期間の開始日を指定した場合と、全期間で InDateRangeSignal によりエントリーを絞った場合とで、
    取引が一致することをテスト（開始日より前に取引がなければ、口座の状態は同じになる）
    """
    with_filter = build_strategy(START, pd.Timestamp("2030-12-31")).run(df)
    with_period = build_strategy().run(df, start=START)

    assert len(_trades(with_period)) > 0, "取引が発生するデータでテストする"
    pd.testing.assert_frame_equal(_trades(with_filter), _trades(with_period))


def test_metrics_use_only_eval_period(df: pd.DataFrame):
    """
    評価期間を指定すると、Portfolio が評価期間の初日から始まり、年率の指標も評価期間だけで計算されることをテスト
    """
    full = build_strategy(START, pd.Timestamp("2030-12-31")).run(df)
    sliced = build_strategy().run(df, start=START)

    assert sliced.wrapper.index[0] == START
    assert sliced.wrapper.index[-1] == df.index[-1]
    full_m, sliced_m = calc_metrics(full), calc_metrics(sliced)
    # 取引は同じなので総リターンは同じ。期間が短い分、CAGR の絶対値とエクスポージャーが大きくなる
    assert sliced_m["total_return"] == pytest.approx(full_m["total_return"])
    assert abs(sliced_m["cagr"]) > abs(full_m["cagr"])
    assert sliced_m["exposure"] > full_m["exposure"]


def test_warmup_data_is_used_on_first_day(df: pd.DataFrame):
    """
    評価期間の初日のシグナル・損切り幅・サイズが、全期間で計算した値と同じ（ウォームアップ期間の情報が入る）
    ことをテスト。切り出してから計算すると、初日は指標が NaN になる
    """
    backtester = build_strategy()
    full = backtester.prepare(df)
    sliced = backtester.prepare(df, start=START, end=END)

    assert sliced.close.index[0] == START
    assert sliced.close.index[-1] <= END
    for attr in ["entries", "exits", "sl_stop", "tp_stop", "size", "min_size"]:
        expected = getattr(full, attr).loc[START:END]
        pd.testing.assert_series_equal(getattr(sliced, attr), expected, check_names=False)
    assert pd.notna(sliced.sl_stop.iloc[0])


def test_future_after_end_does_not_change_result(df: pd.DataFrame):
    """
    評価期間の終了日より後の価格を書き換えても、結果が変わらないことをテスト（未来の情報が入らない）
    """
    changed = df.copy()
    after = changed.index > END
    changed.loc[after, ["Open", "High", "Low", "Close"]] *= 3.0

    base = build_strategy().run(df, start=START, end=END)
    other = build_strategy().run(changed, start=START, end=END)

    assert base.wrapper.index[-1] <= END
    pd.testing.assert_frame_equal(_trades(base), _trades(other))
    pd.testing.assert_series_equal(base.value(), other.value())


def test_without_period_is_unchanged(df: pd.DataFrame):
    """start / end を省略した場合は、全期間で実行される（従来どおり）ことをテスト"""
    portfolio = build_strategy().run(df)
    assert portfolio.wrapper.index[0] == df.index[0]
    assert portfolio.wrapper.index[-1] == df.index[-1]


def test_chart_df_matches_eval_period(df: pd.DataFrame):
    """run() が残す self.df（チャートの出力に使う）も評価期間の行だけになることをテスト"""
    backtester = build_strategy()
    backtester.run(df, start=START, end=END)
    assert backtester.df is not None
    assert backtester.df.index[0] == START
    assert backtester.df.index[-1] <= END
    assert "atr" in backtester.df.columns


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-01", "2023-01-01"),  # 開始日が終了日より後
        ("2040-01-01", None),  # 評価期間にデータがない
    ],
)
def test_invalid_period_raises(df: pd.DataFrame, start, end):
    with pytest.raises(ValueError):
        build_strategy().run(df, start=start, end=end)


def test_warns_when_warmup_is_short(df: pd.DataFrame, caplog: pytest.LogCaptureFixture):
    """評価期間の初日に指標がまだ計算できていない場合に警告することをテスト"""
    with caplog.at_level(logging.WARNING, logger="lib.backtester"):
        build_strategy().run(df, start=df.index[5])
    assert any("ウォームアップ" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lib.backtester"):
        build_strategy().run(df, start=START)
    assert not any("ウォームアップ" in r.getMessage() for r in caplog.records)


def test_multi_backtester_passes_period(df: pd.DataFrame):
    """MultiBacktester の独立モード・資金共有モードにも評価期間が渡ることをテスト"""
    data = {"AAA": df, "BBB": create_random_walk_df(1500, seed=4)}
    mb = MultiBacktester(lambda: build_strategy())

    summary = mb.run_independent(data, start=START, end=END)
    for row in summary.to_dict("records"):
        expected = calc_metrics(build_strategy().run(data[row["ticker"]], start=START, end=END))
        assert row["cagr"] == pytest.approx(expected["cagr"])
        assert row["num_trades"] == expected["num_trades"]

    portfolio = mb.run_portfolio(data, start=START, end=END)
    assert portfolio.wrapper.index[0] == START
    assert portfolio.wrapper.index[-1] <= END
