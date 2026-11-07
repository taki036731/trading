from typing import cast

import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.metrics import calc_metrics, portfolio_stats, year_freq
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_random_walk_df


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
