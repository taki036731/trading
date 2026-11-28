# vectorbt の portfolio.value などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import numpy as np
import pytest

from lib import report
from lib.benchmark import (
    benchmark_metrics,
    buy_and_hold,
    equal_weight_buy_and_hold,
)
from lib.metrics import ACCOUNT_METRIC_COLUMNS, account_metrics, calc_metrics

from .helpers import create_random_walk_df

SLIPPAGE = 0.001
FEES = 0.002


def _hold_return(df, fees: float = 0.0, slippage: float = 0.0) -> float:
    """初日の始値（コスト込み）で買い、最終日の終値で評価したときのリターン"""
    cost = df["Open"].iloc[0] * (1 + slippage) * (1 + fees)
    return df["Close"].iloc[-1] / cost - 1


def test_buy_and_hold_return():
    df = create_random_walk_df(seed=1)
    portfolio = buy_and_hold(df, init_cash=1_000_000.0, fees=FEES, slippage=SLIPPAGE)

    assert portfolio.total_return() == pytest.approx(_hold_return(df, FEES, SLIPPAGE))
    # 買いの1回だけで、最後に売らない
    assert portfolio.orders.count() == 1
    assert portfolio.cash().iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_buy_and_hold_uses_eval_period_only():
    """評価期間より前の価格は使わず、評価期間の初日の始値で買うことをテスト"""
    df = create_random_walk_df(seed=1)
    start, end = df.index[100], df.index[200]
    portfolio = buy_and_hold(df, start=start, end=end, slippage=SLIPPAGE)

    assert portfolio.wrapper.index[0] == start
    assert portfolio.wrapper.index[-1] == end
    expected = _hold_return(df.loc[start:end], slippage=SLIPPAGE)
    assert portfolio.total_return() == pytest.approx(expected)


def test_buy_and_hold_no_data_in_period():
    df = create_random_walk_df(seed=1)
    with pytest.raises(ValueError, match="データがありません"):
        buy_and_hold(df, start=df.index[-1] + np.timedelta64(10, "D"))


def test_equal_weight_buy_and_hold():
    """
    等金額の買い持ちが、各銘柄の買い持ちのリターンの平均になり、
    途中で上場した銘柄の分は上場日まで現金のままになることをテスト
    """
    a = create_random_walk_df(seed=1)
    b = create_random_walk_df(seed=2).iloc[50:]
    init_cash = 1_000_000.0
    portfolio = equal_weight_buy_and_hold(
        {"A": a, "B": b}, init_cash=init_cash, fees=FEES, slippage=SLIPPAGE
    )

    expected = (_hold_return(a, FEES, SLIPPAGE) + _hold_return(b, FEES, SLIPPAGE)) / 2
    assert portfolio.total_return() == pytest.approx(expected)
    cash = portfolio.cash()
    assert cash.loc[a.index[0]] == pytest.approx(init_cash / 2)
    assert cash.loc[b.index[0]] == pytest.approx(0.0, abs=1e-6)
    assert not portfolio.value().isna().any()


def test_account_metrics_matches_calc_metrics():
    """calc_metrics の口座単位の指標が、account_metrics と同じ値になることをテスト"""
    portfolio = buy_and_hold(create_random_walk_df(seed=3))
    metrics = calc_metrics(portfolio)
    account = account_metrics(portfolio)
    assert list(account) == ACCOUNT_METRIC_COLUMNS
    for key, value in account.items():
        assert metrics[key] == pytest.approx(value)
    assert benchmark_metrics(portfolio) == account


def test_benchmark_table():
    df = create_random_walk_df(seed=1)
    strategy = buy_and_hold(df, slippage=SLIPPAGE)
    bench = buy_and_hold(create_random_walk_df(seed=2))
    table = report.benchmark_table(strategy, {"bench": bench})

    assert table.index.tolist() == ACCOUNT_METRIC_COLUMNS
    assert table.columns.tolist() == ["strategy", "bench"]
    assert table.loc["total_return", "bench"] == pytest.approx(bench.total_return())
