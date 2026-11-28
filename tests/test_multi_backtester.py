# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import logging
from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from lib.backtester import SPLIT_BAR_OFFSET, Backtester, PreparedSignals
from lib.benchmark import benchmark_metrics, buy_and_hold
from lib.metrics import calc_metrics
from lib import multi_backtester
from lib.multi_backtester import MultiBacktester, expand_param_grid, load_universe
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import assert_same_daily_result, create_base_df, create_random_walk_df


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

    for row in summary.to_dict("records"):
        single = build_strategy(short=row["short"], tp_mult=row["tp_mult"])
        expected = calc_metrics(single.run(data[row["ticker"]].copy()))
        assert row["total_return"] == pytest.approx(expected["total_return"])
        assert row["num_trades"] == expected["num_trades"]


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
    expected = build_strategy().run(df.copy())
    actual = MultiBacktester(build_strategy).run_portfolio({"AAA": df})
    assert len(expected.trades.records) > 0, "取引が発生するデータでテストする"
    assert_same_daily_result(actual, expected)


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


def test_run_portfolio_uses_costs_from_factory():
    """
    資金共有モードでも、factory が作る Backtester の手数料・スリッページを使うことをテスト
    """

    def factory() -> Backtester:
        backtester = build_manual_strategy()
        backtester.fees = 0.001
        backtester.slippage = 0.01
        return backtester

    df = golden_cross_df(10)
    pf = MultiBacktester(factory).run_portfolio({"AAA": df})

    order = pf.orders.records_readable.iloc[0]
    assert order["Price"] == pytest.approx(df["Open"].iloc[11] * 1.01)
    assert order["Fees"] == pytest.approx(order["Size"] * order["Price"] * 0.001)


def test_slippage_can_be_used_as_grid_parameter():
    """
    factory の引数にすれば、param_grid でスリッページの感度を比較できることをテスト
    """

    def factory(slippage: float) -> Backtester:
        backtester = build_strategy()
        backtester.slippage = slippage
        return backtester

    data = {"AAA": create_random_walk_df(seed=1), "BBB": create_random_walk_df(seed=2)}
    summary = MultiBacktester(factory, {"slippage": [0.0, 0.01]}).run_independent(data)

    returns = summary.groupby("slippage")["total_return"].sum()
    assert summary["num_trades"].sum() > 0, "取引が発生するデータでテストする"
    assert returns[0.01] < returns[0.0]


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


def build_narrow_stop_strategy() -> Backtester:
    """エントリー当日にラインへ届く日が多く出るように、SL/TP の幅を狭くした戦略"""
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(1),
        sl=AtrStopLoss(0.5),
        position_sizer=RiskPercentageSizer(0.02),
    )


def test_calc_metrics_annualizes_by_calendar_days_with_split_bars():
    """
    エントリー当日に足を分けて行が増えても、CAGR が暦日ベースの年率のままであることをテスト
    """
    df = create_random_walk_df(days=1000, seed=4)
    pf = build_narrow_stop_strategy().run(df.copy())
    assert len(pf.wrapper.index) > len(df), "足を分ける日が出るデータでテストする"

    metrics = calc_metrics(pf)
    years = (df.index[-1] - df.index[0]).days / 365.25
    expected = (1 + metrics["total_return"]) ** (1 / years) - 1
    assert metrics["cagr"] == pytest.approx(expected, rel=1e-3)


def test_run_portfolio_single_ticker_matches_backtester_with_split_bars():
    """
    足を分ける日があっても、資金共有モードの1銘柄は Backtester.run と同じ資産推移になることをテスト
    """
    df = create_random_walk_df(days=500, seed=3)
    expected = build_narrow_stop_strategy().run(df.copy())
    assert len(expected.wrapper.index) > len(df), "足を分ける日が出るデータでテストする"

    actual = MultiBacktester(build_narrow_stop_strategy).run_portfolio({"AAA": df})
    assert_same_daily_result(actual, expected)


def test_run_portfolio_stops_on_entry_day():
    """
    資金共有モードでも、エントリー当日に SL ラインを割った銘柄はその日のうちに決済されることをテスト。
    あわせて、その決済の現金は同じ日の始値の買い（他の銘柄）には使えないことを確認する
    """
    touched = golden_cross_df(10)
    entry_day = touched.index[11]
    touched.loc[entry_day, "Low"] = touched.loc[entry_day, "Open"] - 100  # SL ライン（2ATR=40 下）を割る

    pf = MultiBacktester(build_manual_strategy).run_portfolio(
        {"AAA": touched, "BBB": golden_cross_df(10)}
    )
    orders = pf.orders.records_readable

    aaa = orders[orders["Column"] == "AAA"]
    assert aaa["Side"].tolist() == ["Buy", "Sell"]
    assert aaa["Timestamp"].tolist() == [entry_day, entry_day + SPLIT_BAR_OFFSET]

    # BBB の買いは、AAA の当日の決済がない場合と同じサイズ（同じ日の始値の時点では現金が戻っていない）
    baseline = MultiBacktester(build_manual_strategy).run_portfolio(
        {"AAA": golden_cross_df(10), "BBB": golden_cross_df(10)}
    ).orders.records_readable
    bbb = orders[orders["Column"] == "BBB"]
    assert bbb["Size"].tolist() == baseline[baseline["Column"] == "BBB"]["Size"].tolist()

    # 分けた行でも、口座の資産は NaN にならない
    assert not pf.value().isna().any()


def build_manual_strategy_with_dead_cross() -> Backtester:
    """build_manual_strategy に、デッドクロスでのエグジット（始値で約定）を足した戦略"""
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )


# AAA の買い（約10万円）の後、BBB の 1単元（約10万円）を買うには現金が足りず、
# AAA の売却代金が戻れば買える初期資金
CASH_FOR_ONE_LOT = 250_000.0


def _run_after_aaa_exit(
    aaa,
    bbb_cross_day: int,
    factory: Callable[[], Backtester] = build_manual_strategy,
):
    """AAA が 6日目に買い、BBB が bbb_cross_day + 1 日目に買おうとする資金共有の Portfolio"""
    return MultiBacktester(factory).run_portfolio(
        {"AAA": aaa, "BBB": golden_cross_df(bbb_cross_day)}, init_cash=CASH_FOR_ONE_LOT
    )


def _orders(pf, column: str):
    orders = pf.orders.records_readable
    return orders[orders["Column"] == column]


def test_run_portfolio_intraday_stop_cash_not_used_for_same_day_open_buy():
    """
    場中の SL の決済の代金は、同じ日の始値の買いには使えず、翌日の買いから使えることをテスト（Issue #36）
    """
    aaa = golden_cross_df(5)
    stop_day = aaa.index[11]
    aaa.loc[stop_day, "Low"] = aaa.loc[stop_day, "Open"] - 100  # 場中に SL ライン（2ATR=40 下）を割る

    # 同じ日の始値の買い: AAA の売却代金はまだ使えず、現金不足で買えない
    same_day = _run_after_aaa_exit(aaa, bbb_cross_day=10)
    sell = _orders(same_day, "AAA").iloc[-1]
    assert (sell["Side"], sell["Timestamp"]) == ("Sell", stop_day + SPLIT_BAR_OFFSET)
    assert _orders(same_day, "BBB").empty

    # 翌日の始値の買い: AAA の売却代金で買える
    next_day = _run_after_aaa_exit(aaa, bbb_cross_day=11)
    bbb = _orders(next_day, "BBB")
    assert bbb["Timestamp"].tolist() == [aaa.index[12]]
    assert bbb["Size"].tolist() == [100]


def test_run_portfolio_gap_stop_cash_not_used_for_same_day_open_buy():
    """
    寄り付きのギャップで SL ラインを越えた決済（始値で約定）の代金も、同じ日の始値の買いには使えないことをテスト
    """
    aaa = golden_cross_df(5)
    stop_day = aaa.index[11]
    aaa.loc[stop_day, ["Open", "Low"]] = aaa.loc[stop_day, "Open"] - 100  # 始値で SL ラインを割る

    pf = _run_after_aaa_exit(aaa, bbb_cross_day=10)
    sell = _orders(pf, "AAA").iloc[-1]
    assert (sell["Side"], sell["Timestamp"]) == ("Sell", stop_day + SPLIT_BAR_OFFSET)
    # 約定価格は始値（スリッページ込み）で、分けない場合と同じ
    assert sell["Price"] == pytest.approx(aaa.loc[stop_day, "Open"] * (1 - 0.001))
    assert _orders(pf, "BBB").empty


def test_run_portfolio_signal_exit_cash_used_for_same_day_open_buy():
    """
    シグナルの売り（始値で約定）の代金は、同じ日の始値の買いに使えることをテスト
    """
    aaa = golden_cross_df(5)
    aaa.loc[aaa.index[10:], "short_ma"] = 990  # 10日目にデッドクロス → 11日目の始値で売る

    pf = _run_after_aaa_exit(aaa, bbb_cross_day=10, factory=build_manual_strategy_with_dead_cross)
    sell = _orders(pf, "AAA").iloc[-1]
    assert (sell["Side"], sell["Timestamp"]) == ("Sell", aaa.index[11])
    bbb = _orders(pf, "BBB")
    assert bbb["Timestamp"].tolist() == [aaa.index[11]]
    assert bbb["Size"].tolist() == [100]


def test_split_bars_for_cash_sharing_first_bar_is_previous_close():
    """
    資金共有用に分けた足の1行目は前日終値だけの足で、シグナル・サイズは1行目に残ることをテスト
    """
    df = golden_cross_df(5)
    prepared = build_manual_strategy().prepare(df.copy())
    day = df.index[11]
    split = prepared.split_bars_for_cash_sharing(pd.DatetimeIndex([day]))

    prev_close = df["Close"].iloc[10]
    for attr in ["open", "high", "low", "close"]:
        assert getattr(split, attr)[day] == prev_close
        assert getattr(split, attr)[day + SPLIT_BAR_OFFSET] == df[attr.capitalize()].iloc[11]
    assert split.price[day] == prepared.price[day]
    assert split.entries[day] == prepared.entries[day]
    assert not split.entries[day + SPLIT_BAR_OFFSET]
    assert len(split.close) == len(df) + 1


def test_aggregate_by_attribute():
    data = {
        "AAA": create_random_walk_df(seed=1),
        "BBB": create_random_walk_df(seed=2),
        "CCC": create_random_walk_df(seed=3),
    }
    mb = MultiBacktester(build_strategy, {"short": [5, 10]})
    summary = mb.run_independent(data)
    summary["size"] = summary["ticker"].map({"AAA": "Large", "BBB": "Small", "CCC": "Small"})
    agg = mb.aggregate(summary, by=["size"])

    assert agg[["short", "size"]].values.tolist() == [
        [5, "Large"],
        [5, "Small"],
        [10, "Large"],
        [10, "Small"],
    ]
    small = agg[(agg["short"] == 5) & (agg["size"] == "Small")].iloc[0]
    group = summary[(summary["short"] == 5) & (summary["size"] == "Small")]
    assert small["num_tickers"] == 2
    assert small["total_return_median"] == pytest.approx(group["total_return"].median())


def test_aggregate_by_missing_column():
    mb = MultiBacktester(build_strategy)
    summary = mb.run_independent({"AAA": create_random_walk_df(seed=1)})
    with pytest.raises(ValueError, match="sector"):
        mb.aggregate(summary, by=["sector"])


def test_load_universe_skips_failed_tickers(monkeypatch, caplog):
    """
    空のデータや取得時の例外の銘柄はスキップし、最後に件数をまとめて警告することをテスト
    """
    df = create_random_walk_df(seed=1)

    def fake_fetch(ticker, start=None, end=None, source=None, **kwargs):
        if ticker == "ERR.T":
            raise ConnectionError("network")
        if ticker == "EMPTY.T":
            return pd.DataFrame()
        return df

    monkeypatch.setattr(multi_backtester.dl, "fetch_stock_data", fake_fetch)
    with caplog.at_level(logging.WARNING, logger="lib.multi_backtester"):
        data = load_universe(["AAA.T", "ERR.T", "EMPTY.T", "BBB.T"])

    assert list(data) == ["AAA.T", "BBB.T"]
    assert "4 銘柄のうち 2 銘柄" in caplog.text
    assert "ERR.T" in caplog.text and "EMPTY.T" in caplog.text


def test_load_universe_passes_mode_and_logs_last_dates(monkeypatch, caplog):
    """
    取得モードと有効期限を fetch_stock_data に渡し、データの最終日の範囲をログに出すことをテスト
    """
    df = create_random_walk_df(seed=1)
    calls = []

    def fake_fetch(ticker, start=None, end=None, source=None, **kwargs):
        calls.append(kwargs)
        return df.iloc[:-5] if ticker == "OLD.T" else df

    monkeypatch.setattr(multi_backtester.dl, "fetch_stock_data", fake_fetch)
    with caplog.at_level(logging.INFO, logger="lib.multi_backtester"):
        load_universe(["OLD.T", "NEW.T"], mode="offline", max_age_days=3)

    assert calls == [{"mode": "offline", "max_age_days": 3}] * 2
    assert f"最古 {df.index[-6]:%Y-%m-%d}（OLD.T）" in caplog.text
    assert f"最新 {df.index[-1]:%Y-%m-%d}（NEW.T）" in caplog.text


def test_run_independent_adds_buy_and_hold_columns():
    """
    run_independent に、同じ評価期間・初期資金・取引コストでの買い持ちの列と、
    戦略との CAGR の差（excess_cagr）が入ることをテスト
    """
    data = {"AAA": create_random_walk_df(seed=1), "BBB": create_random_walk_df(seed=2)}
    start = data["AAA"].index[60]
    mb = MultiBacktester(build_strategy, {"short": [5, 10]})
    summary = mb.run_independent(data, start=start)

    reference = build_strategy()
    for ticker, df in data.items():
        bh = benchmark_metrics(
            buy_and_hold(df, start=start, init_cash=reference.init_cash, **reference.cost_params)
        )
        rows = summary[summary["ticker"] == ticker]
        assert (rows["bh_cagr"] == bh["cagr"]).all()
        assert (rows["bh_max_drawdown"] == bh["max_drawdown"]).all()
        np.testing.assert_allclose(rows["excess_cagr"], rows["cagr"] - bh["cagr"])

    agg = mb.aggregate(summary)
    for short, group in summary.groupby("short"):
        row = agg[agg["short"] == short].iloc[0]
        traded = group[(group["num_trades"] + group["open_trades"]) > 0]
        assert row["beat_bh_ratio"] == pytest.approx((traded["excess_cagr"] > 0).mean())
        assert row["excess_cagr_median"] == pytest.approx(traded["excess_cagr"].median())


def test_run_independent_without_benchmark():
    mb = MultiBacktester(build_strategy)
    summary = mb.run_independent({"AAA": create_random_walk_df(seed=1)}, benchmark=False)
    assert "bh_cagr" not in summary.columns
    assert "beat_bh_ratio" not in mb.aggregate(summary).columns
