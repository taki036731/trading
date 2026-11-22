from typing import cast

import numpy as np
import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.metrics import calc_metrics, closed_and_open_trades, portfolio_stats, year_freq
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df, create_random_walk_df


def build_strategy(sl_mult: float = 2, tp_mult: float = 4) -> Backtester:
    """テスト用の戦略（EMA 5/20 のクロス、ATR ベースの SL/TP、リスク割合サイジング）"""
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(tp_mult),
        sl=AtrStopLoss(sl_mult),
        position_sizer=RiskPercentageSizer(0.02),
    )


def test_year_freq_by_calendar_days():
    """
    1年あたりの本数が、行数 ÷ 暦年で求められることをテスト
    """
    index = pd.bdate_range("2020-01-01", periods=1000)
    years = (index[-1] - index[0]).days / 365.25
    assert year_freq(index) == f"{1000 / years} days"


@pytest.mark.parametrize("periods", [0, 1])
def test_year_freq_too_short(periods: int):
    """
    期間が短すぎる場合は None（vectorbt の既定値を使う）になることをテスト
    """
    assert year_freq(pd.bdate_range("2020-01-01", periods=periods)) is None


@pytest.mark.parametrize(
    ("sl_mult", "tp_mult"),
    [
        (2, 4),
        # エントリー当日にラインへ届き、足を2行に分ける日が多く出る設定
        (0.5, 1),
    ],
)
def test_portfolio_stats_matches_calc_metrics(sl_mult: float, tp_mult: float):
    """
    portfolio_stats の年率の指標が、calc_metrics と同じ基準で換算されることをテスト
    """
    df = create_random_walk_df(days=1000, seed=4)
    pf = build_strategy(sl_mult, tp_mult).run(df.copy())
    if sl_mult < 1:
        assert len(pf.wrapper.index) > len(df), "足を分ける日が出るデータでテストする"

    stats = portfolio_stats(pf)
    metrics = calc_metrics(pf)
    assert stats["Sharpe Ratio"] == pytest.approx(metrics["sharpe"])
    assert stats["Sortino Ratio"] == pytest.approx(metrics["sortino"])
    # 年率換算しない指標は、そのまま
    assert stats["Total Return [%]"] == pytest.approx(metrics["total_return"] * 100)


def test_portfolio_stats_differs_from_default_365_days():
    """
    営業日だけの日足では、vectorbt 既定（1年=365本）の stats() と年率の指標が異なることをテスト
    """
    df = create_random_walk_df(days=1000, seed=4)
    pf = build_strategy().run(df.copy())
    default_stats = cast(pd.Series, pf.stats())
    assert portfolio_stats(pf)["Sharpe Ratio"] != pytest.approx(
        default_stats["Sharpe Ratio"]
    )


def build_base_df_strategy() -> Backtester:
    """create_base_df の列をそのまま使う戦略（利確させず、損切りにも届かない）"""
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(100),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


def open_at_end_df(closed_first: bool) -> pd.DataFrame:
    """
    最後にゴールデンクロスし、保有したまま期間が終わるデータ。
    closed_first=True なら、その前に決済済みのトレード（10日目に買いシグナル、20日目に売りシグナル）が1回ある
    """
    df = create_base_df(days=40)
    if closed_first:
        df.loc[df.index[10:20], "short_ma"] = 1010
    df.loc[df.index[30:], "short_ma"] = 1010
    return df


def test_open_trade_excluded_from_trade_metrics():
    """
    期間の最後に保有中のトレードは、num_trades / win_rate などに入らず、open_trades に数えられる
    """
    pf = build_base_df_strategy().run(open_at_end_df(closed_first=True))
    closed, open_ = closed_and_open_trades(pf)
    assert (closed.count(), open_.count()) == (1, 1), "決済済みと保有中のトレードが1回ずつあるデータでテストする"

    metrics = calc_metrics(pf)

    assert metrics["num_trades"] == 1
    assert metrics["open_trades"] == 1
    assert metrics["win_rate"] == pytest.approx(closed.win_rate())
    assert metrics["expectancy"] == pytest.approx(closed.expectancy())
    assert metrics["avg_trade_return"] == pytest.approx(closed.returns.mean())  # pyright: ignore[reportAttributeAccessIssue]
    # 口座単位の指標は、保有中のトレードの含み損益も含む
    assert metrics["total_return"] == pytest.approx(pf.total_return())


def test_only_open_trade_gives_nan_trade_metrics():
    """決済済みのトレードがなければ、トレード単位の指標は NaN（保有中のトレードは open_trades に数える）"""
    metrics = calc_metrics(build_base_df_strategy().run(open_at_end_df(closed_first=False)))

    assert metrics["num_trades"] == 0
    assert metrics["open_trades"] == 1
    for key in ["win_rate", "profit_factor", "expectancy", "avg_trade_return"]:
        assert np.isnan(metrics[key]), key


def test_trade_metrics_match_portfolio_stats():
    """
    calc_metrics のトレード単位の指標が、portfolio_stats（決済済みだけで計算する）と一致することをテスト
    """
    # 期間の最後に保有中のトレードが残るデータ
    pf = build_strategy().run(create_random_walk_df(days=1000, seed=9))
    stats = portfolio_stats(pf)
    metrics = calc_metrics(pf)
    assert stats["Total Open Trades"] == 1, "保有中のトレードが残るデータでテストする"

    assert metrics["num_trades"] == stats["Total Closed Trades"]
    assert metrics["open_trades"] == stats["Total Open Trades"]
    assert metrics["win_rate"] * 100 == pytest.approx(stats["Win Rate [%]"])
    assert metrics["profit_factor"] == pytest.approx(stats["Profit Factor"])
    assert metrics["expectancy"] == pytest.approx(stats["Expectancy"])


def test_aggregate_counts_ticker_with_only_open_trade():
    """
    保有中のトレードだけの銘柄も、取引ありとして total_return の集計に入る（win_rate は NaN なので外れる）
    """
    mb = MultiBacktester(build_base_df_strategy)
    summary = mb.run_independent(
        {"OPEN": open_at_end_df(closed_first=False), "BOTH": open_at_end_df(closed_first=True)}
    )
    assert summary["open_trades"].tolist() == [1, 1]
    assert summary["num_trades"].tolist() == [0, 1]

    row = mb.aggregate(summary).iloc[0]

    assert row["num_traded_tickers"] == 2
    assert row["zero_trade_tickers"] == 0
    assert row["total_return_median"] == pytest.approx(summary["total_return"].median())
    # win_rate は決済済みのトレードがある銘柄だけ
    assert row["win_rate_median"] == pytest.approx(summary["win_rate"].iloc[1])
    assert row["total_trades"] == 1
    assert row["total_open_trades"] == 2
