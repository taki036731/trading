from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from lib.backtester import Backtester, PreparedSignals
from lib.multi_backtester import MultiBacktester, calc_metrics, expand_param_grid
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df, create_random_walk_df


def build_strategy(short: int = 5, long: int = 20, tp_mult: float = 4) -> Backtester:
    """テスト用の戦略（strategy01 から期間・ATR フィルターを除いたもの）"""
    return Backtester(
        indicators=[MAIndicator("EMA", short, long), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(tp_mult),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


def build_manual_strategy() -> Backtester:
    """create_base_df の列をそのまま使う（インジケーター計算なし）戦略"""
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


def golden_cross_df(cross_day: int, days: int = 30, start_date: str = "2024-01-01"):
    """cross_day 日目にゴールデンクロスが起きる（cross_day + 1 日目にエントリー）データ"""
    df = create_base_df(days=days, start_date=start_date)
    df.loc[df.index[cross_day:], "short_ma"] = 1010
    return df


def test_expand_param_grid():
    assert expand_param_grid(None) == [{}]
    assert expand_param_grid({"short": [5, 10], "long": [20, 50]}) == [
        {"short": 5, "long": 20},
        {"short": 5, "long": 50},
        {"short": 10, "long": 20},
        {"short": 10, "long": 50},
    ]


def test_run_independent_matches_single_backtester():
    """
    独立モードの各行が、Backtester.run を単体で実行した結果と一致することをテスト
    """
    data = {
        "AAA": create_random_walk_df(seed=1),
        "BBB": create_random_walk_df(seed=2),
    }
    grid = {"short": [5, 10], "tp_mult": [3, 4]}
    summary = MultiBacktester(build_strategy, grid).run_independent(data)

    assert len(summary) == 2 * 4
    assert summary["num_trades"].sum() > 0, "取引が発生するデータでテストする"

    for row in summary.itertuples():
        single = build_strategy(short=row.short, tp_mult=row.tp_mult)
        expected = calc_metrics(single.run(data[row.ticker].copy()))
        assert row.total_return == pytest.approx(expected["total_return"])
        assert row.num_trades == expected["num_trades"]


def test_run_independent_does_not_modify_input():
    """
    indicator が列を追加しても、渡した元の DataFrame は変わらないことをテスト
    """
    df = create_random_walk_df(seed=1)
    columns = df.columns.tolist()
    MultiBacktester(build_strategy).run_independent({"AAA": df})
    assert df.columns.tolist() == columns


def test_aggregate_by_params():
    data = {
        "AAA": create_random_walk_df(seed=1),
        "BBB": create_random_walk_df(seed=2),
        "CCC": create_random_walk_df(seed=3),
    }
    mb = MultiBacktester(build_strategy, {"short": [5, 10]})
    summary = mb.run_independent(data)
    agg = mb.aggregate(summary)

    assert agg["short"].tolist() == [5, 10]
    assert (agg["num_tickers"] == 3).all()
    for short, group in summary.groupby("short"):
        row = agg[agg["short"] == short].iloc[0]
        assert row["total_return_median"] == pytest.approx(group["total_return"].median())
        assert row["positive_ratio"] == pytest.approx((group["total_return"] > 0).mean())
        assert row["total_trades"] == group["num_trades"].sum()


def test_run_portfolio_single_ticker_matches_backtester():
    """
    資金共有モードで1銘柄だけ渡した場合、Backtester.run と同じ資産推移になることをテスト
    """
    df = create_random_walk_df(seed=1)
    expected = build_strategy().run(df.copy()).value()
    actual = MultiBacktester(build_strategy).run_portfolio({"AAA": df}).value()
    np.testing.assert_allclose(actual.values, expected.values)


def test_run_portfolio_shares_cash():
    """
    同じ日に2銘柄でエントリーした場合、2銘柄目は残りの現金を基準にサイズが決まることをテスト
    """
    data = {"AAA": golden_cross_df(10), "BBB": golden_cross_df(10)}
    init_cash = 1_000_000.0
    pf = MultiBacktester(build_manual_strategy).run_portfolio(data, init_cash=init_cash)

    orders = pf.orders.records_readable
    assert len(orders) == 2
    # 先に処理されるのは渡した順の1銘柄目
    first, second = orders.iloc[0], orders.iloc[1]
    assert first["Column"] == "AAA"
    assert second["Column"] == "BBB"
    assert (first["Timestamp"], second["Timestamp"]) == (data["AAA"].index[11],) * 2

    # 2銘柄目は、1銘柄目の購入後の現金に対する割合で買う（現金基準）
    pct = 0.02 / (2 * 20.0 / data["AAA"]["Close"].iloc[10])  # risk / sl_pct
    cash_after_first = init_cash - first["Size"] * first["Price"]
    expected_second = np.floor(cash_after_first * pct / second["Price"] / 100) * 100
    assert second["Size"] == expected_second
    assert second["Size"] < first["Size"]

    # 2銘柄で1つの口座なので、資産推移は1本
    assert pf.value().ndim == 1
    assert pf.value().iloc[0] == init_cash


def test_run_portfolio_no_orders_on_missing_days():
    """
    上場前や欠損日など、データのない日に注文（SL/TP を含む）が出ないことをテスト
    """
    early = golden_cross_df(5, days=40, start_date="2024-01-01")
    # BBB は 20日遅れでデータが始まり、データ上の1日目のクロスで2日目にエントリーする
    late = golden_cross_df(1, days=20, start_date="2024-01-21")
    # AAA の保有中に、AAA だけデータのない日を作る（BBB にはデータがある）
    gap_day = early.index[25]
    early = early.drop(gap_day)

    pf = MultiBacktester(build_manual_strategy).run_portfolio({"AAA": early, "BBB": late})
    orders = pf.orders.records_readable

    for _, order in orders.iterrows():
        source = early if order["Column"] == "AAA" else late
        assert order["Timestamp"] in source.index

    assert gap_day not in orders["Timestamp"].tolist()

    bbb = orders[orders["Column"] == "BBB"]
    assert len(bbb) == 1
    assert bbb["Timestamp"].iloc[0] == late.index[2]
    # BBB のデータがない期間も、口座の資産は NaN にならない
    assert not pf.value().isna().any()


def test_combine_keys_match_prepared_signals():
    """
    PreparedSignals の属性、to_vbt_kwargs() のキー、_combine() のキーが一致することをテスト
    （PreparedSignals に属性を足したとき、to_vbt_kwargs() や _combine() の更新漏れを検出する）
    """
    data = {"AAA": golden_cross_df(10), "BBB": golden_cross_df(10)}
    prepared = {t: build_manual_strategy().prepare(df.copy()) for t, df in data.items()}

    attrs = {f.name for f in fields(PreparedSignals)}
    assert set(prepared["AAA"].to_vbt_kwargs()) == attrs
    assert set(MultiBacktester._combine(prepared)) == attrs


def test_run_portfolio_empty_data():
    with pytest.raises(ValueError):
        MultiBacktester(build_strategy).run_portfolio({})



def test_calc_metrics_annualizes_by_business_days():
    """
    営業日だけの日足でも、CAGR が暦日ベースの年率で計算されることをテスト
    """
    df = create_random_walk_df(days=1000, seed=4)
    pf = build_strategy().run(df.copy())
    metrics = calc_metrics(pf)
    years = (df.index[-1] - df.index[0]).days / 365.25
    expected = (1 + metrics["total_return"]) ** (1 / years) - 1
    assert metrics["cagr"] == pytest.approx(expected, rel=1e-3)
