# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import logging

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from lib.backtester import JP_MARKET_PARAMS, Backtester
from lib.metrics import calc_metrics, entry_order_stats, warn_low_fill_rate
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df, create_random_walk_df


def build_strategy(
    init_cash: float = 1000000.0, min_fill_rate: float | None = 0.8
) -> Backtester:
    """
    create_base_df の列をそのまま使う戦略。
    損切り幅は 2ATR=40円（約4%）なので、サイズは現金の約50%になる
    """
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=init_cash,
        min_fill_rate=min_fill_rate,
    )


def build_random_walk_strategy(init_cash: float = 1000000.0) -> Backtester:
    """インジケーター計算から通しで動かす戦略（ランダムウォークのデータ用）"""
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=init_cash,
    )


def cross_df() -> pd.DataFrame:
    """10日目にゴールデンクロス（11日目にエントリー）、20日目にデッドクロスが起きるデータ"""
    df = create_base_df(days=30)
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[df.index[20:], "short_ma"] = 990
    return df


def test_filled_entry_is_counted():
    """
    資金100万円の約50%（約50万円）で、1単元（約10万円）を買える場合は約定として数える
    """
    stats = entry_order_stats(build_strategy().run(cross_df()))

    assert stats["entry_orders"].iloc[0] == 1
    assert stats["entry_filled"].iloc[0] == 1
    assert stats["entry_dropped"].iloc[0] == 0
    assert stats["entry_fill_rate"].iloc[0] == 1.0


def test_entry_below_one_lot_is_counted_as_dropped():
    """
    資金10万円の約50%（約5万円）では1単元（約10万円）に届かず、注文が捨てられる。
    取引履歴には残らないが、捨てられた数と理由として数える
    """
    portfolio = build_strategy(100000.0).run(cross_df())
    stats = entry_order_stats(portfolio)

    assert len(portfolio.trades.records) == 0
    assert stats["entry_orders"].iloc[0] == 1
    assert stats["entry_dropped"].iloc[0] == 1
    assert stats["entry_fill_rate"].iloc[0] == 0.0
    # 割合（Percent）のサイズは MinSizeNotReached になる。
    # 株数（Amount）のサイズでは、size_granularity で先に0株へ切り捨てられて SizeZero になる
    assert stats["dropped_MinSizeNotReached"].iloc[0] == 1


def test_signal_while_holding_is_not_counted():
    """
    保有中に出たエントリーシグナルは注文にならないため、注文の数に含めない
    """
    close = pd.Series([10.0, 11, 12, 13, 14, 15])
    entries = pd.Series([True, True, False, True, False, False])
    exits = pd.Series([False, False, True, False, False, False])
    size = pd.Series([1000.0, 1000, 1000, 50, 1000, 1000])
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        size=size,
        size_type="amount",
        size_granularity=100,
        min_size=100,
        init_cash=1e6,
        log=True,
        freq="D",
    )
    stats = entry_order_stats(portfolio)

    # 0日目（約定）と3日目（50株で捨てられる）だけ。1日目は保有中のため注文にならない
    assert stats["entry_orders"].iloc[0] == 2
    assert stats["entry_filled"].iloc[0] == 1
    assert stats["entry_dropped"].iloc[0] == 1


def test_no_entry_order_gives_nan_fill_rate():
    """エントリー注文がなければ、約定率は NaN（0% と区別する）"""
    df = create_base_df(days=30)  # クロスが起きない
    metrics = calc_metrics(build_strategy().run(df))

    assert metrics["entry_orders"] == 0
    assert metrics["entry_dropped"] == 0
    assert np.isnan(metrics["entry_fill_rate"])


def test_calc_metrics_includes_entry_columns():
    metrics = calc_metrics(build_strategy(100000.0).run(cross_df()))

    assert metrics["entry_orders"] == 1
    assert metrics["entry_dropped"] == 1
    assert metrics["entry_fill_rate"] == 0.0


def test_logging_does_not_change_results():
    """
    log=True（JP_MARKET_PARAMS）は注文の記録を残すだけで、売買の結果は変わらないことをテスト
    """
    df = create_random_walk_df(days=1000, seed=4)
    backtester = build_random_walk_strategy()
    with_log = backtester.run(df.copy())

    kwargs = backtester.prepare(df.copy()).split_entry_day_bars(backtester.slippage)
    without_log = vbt.Portfolio.from_signals(
        **kwargs.to_vbt_kwargs(),
        init_cash=backtester.init_cash,
        **backtester.cost_params,
        **{**JP_MARKET_PARAMS, "log": False},
    )

    assert len(with_log.orders.records) > 0, "取引が発生するデータでテストする"
    pd.testing.assert_frame_equal(
        with_log.orders.records_readable, without_log.orders.records_readable
    )
    np.testing.assert_allclose(with_log.value().values, without_log.value().values)


def test_warns_when_fill_rate_is_low(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="lib.metrics"):
        build_strategy(100000.0).run(cross_df(), name="AAA")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "AAA" in message
    assert "0%" in message


def test_no_warning_when_fill_rate_is_high(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="lib.metrics"):
        build_strategy().run(cross_df())

    assert not caplog.records


@pytest.mark.parametrize(("min_fill_rate", "warned"), [(None, False), (0.0, False), (1.0, True)])
def test_min_fill_rate_can_be_changed(
    caplog: pytest.LogCaptureFixture, min_fill_rate: float | None, warned: bool
):
    """しきい値は Backtester の引数で変えられ、None で警告を止められる"""
    stats = pd.DataFrame(
        {
            "entry_orders": [4],
            "entry_filled": [3],
            "entry_dropped": [1],
            "entry_fill_rate": [0.75],
            "dropped_SizeZero": [1],
        },
        index=["AAA"],
    )
    with caplog.at_level(logging.WARNING, logger="lib.metrics"):
        warn_low_fill_rate(stats, min_fill_rate)

    assert bool(caplog.records) == warned


def test_run_independent_reports_dropped_entries():
    data = {"AAA": cross_df()}
    summary = MultiBacktester(lambda: build_strategy(100000.0)).run_independent(data)

    assert summary["entry_orders"].tolist() == [1]
    assert summary["entry_dropped"].tolist() == [1]
    assert summary["entry_fill_rate"].tolist() == [0.0]


def test_aggregate_excludes_zero_trade_tickers():
    """
    取引が0回の銘柄は、positive_ratio と指標の中央値・平均から除外し、件数を列に出す
    """
    data = {
        "AAA": create_random_walk_df(seed=1),
        "BBB": create_random_walk_df(seed=2),
        # クロスが起きず、取引が0回。指標はインジケーターで計算するので、あらかじめ入っている列は除く
        "ZERO": create_base_df(days=300).drop(columns=["short_ma", "long_ma", "atr"]),
    }
    mb = MultiBacktester(build_random_walk_strategy)
    summary = mb.run_independent(data)
    assert summary.loc[summary["ticker"] == "ZERO", "num_trades"].item() == 0
    traded = summary[summary["ticker"] != "ZERO"]
    assert (traded["num_trades"] > 0).all(), "取引が発生するデータでテストする"

    row = mb.aggregate(summary).iloc[0]

    assert row["num_tickers"] == 3
    assert row["num_traded_tickers"] == 2
    assert row["zero_trade_tickers"] == 1
    assert row["positive_ratio"] == pytest.approx((traded["total_return"] > 0).mean())
    assert row["total_return_median"] == pytest.approx(traded["total_return"].median())
    assert row["fill_rate_median"] == pytest.approx(summary["entry_fill_rate"].median())
    assert row["total_entry_dropped"] == summary["entry_dropped"].sum()


def test_aggregate_all_zero_trade_tickers_gives_nan():
    """全銘柄で取引が0回なら、positive_ratio は 0 ではなく NaN"""
    mb = MultiBacktester(lambda: build_strategy(100000.0))
    summary = mb.run_independent({"AAA": cross_df(), "BBB": cross_df()})

    row = mb.aggregate(summary).iloc[0]

    assert row["num_traded_tickers"] == 0
    assert np.isnan(row["positive_ratio"])
    assert np.isnan(row["total_return_median"])


def test_run_portfolio_warns_per_ticker(caplog: pytest.LogCaptureFixture):
    """資金共有モードでも、約定率が低い銘柄を銘柄名付きで警告する"""
    with caplog.at_level(logging.WARNING, logger="lib.metrics"):
        portfolio = MultiBacktester(lambda: build_strategy(100000.0)).run_portfolio(
            {"AAA": cross_df(), "BBB": cross_df()}
        )

    stats = entry_order_stats(portfolio)
    assert stats.index.tolist() == ["AAA", "BBB"]
    assert stats["entry_dropped"].tolist() == [1, 1]
    messages = [r.getMessage() for r in caplog.records]
    assert any("AAA" in m for m in messages)
    assert any("BBB" in m for m in messages)
