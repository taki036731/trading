# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

from typing import cast

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from lib.backtester import JP_MARKET_PARAMS, SPLIT_BAR_OFFSET, Backtester
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df, create_random_walk_df

# index=10 でゴールデンクロス -> index=11 の始値 1000 でエントリー
ENTRY = 11


def _create_flat_df() -> pd.DataFrame:
    """
    値動きのない DataFrame（O=C=1000, H=1010, L=990, ATR=20）。
    index=11 の始値 1000 でエントリーし、損切り幅は 2ATR / 1000 = 4%（初期の SL ライン 960）。
    """
    df = create_base_df(days=30)
    df["Open"] = 1000.0
    df["Close"] = 1000.0
    df["High"] = 1010.0
    df["Low"] = 990.0
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[:, "long_ma"] = 1000
    return df


def _set_bar(df: pd.DataFrame, i: int, o: float, h: float, low: float, c: float) -> None:
    df.loc[df.index[i], ["Open", "High", "Low", "Close"]] = [o, h, low, c]


def _build(is_trailing: bool) -> Backtester:
    """2ATR の損切りだけで決済する戦略（利確・エグジットシグナルなし、コストなし）"""
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2, is_trailing=is_trailing),
        position_sizer=RiskPercentageSizer(0.02),
        fees=0.0,
        slippage=0.0,
    )


def _create_rise_then_fall_df() -> pd.DataFrame:
    """エントリー後に高値 1100 まで上がり、index=14 に 1050 まで下がるデータ"""
    df = _create_flat_df()
    _set_bar(df, ENTRY + 1, 1050, 1060, 1040, 1050)
    _set_bar(df, ENTRY + 2, 1090, 1100, 1080, 1090)
    _set_bar(df, ENTRY + 3, 1080, 1085, 1050, 1080)  # 1100 × 0.96 = 1056 を割る
    for i in range(ENTRY + 4, len(df)):
        _set_bar(df, i, 1080, 1090, 1070, 1080)
    return df


def test_trailing_stop_follows_high():
    """
    トレーリングでは、SL ラインが保有中の最高値 × (1 - 損切り幅) まで上がって決済されることをテスト
    """
    df = _create_rise_then_fall_df()

    trades = _build(is_trailing=True).run(df.copy()).trades.records_readable
    assert len(trades) == 1
    assert trades["Entry Timestamp"].iloc[0] == df.index[ENTRY]
    assert trades["Exit Timestamp"].iloc[0] == df.index[ENTRY + 3]
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(1100 * 0.96)

    # トレーリングなしでは、初期の SL ライン 960 に届かないので保有したまま
    trades = _build(is_trailing=False).run(df.copy()).trades.records_readable
    assert len(trades) == 1
    assert trades["Status"].iloc[0] == "Open"


def test_trailing_high_takes_effect_from_next_bar():
    """
    当日の高値で上がったラインは翌日から効き、高値を更新した足ではそのラインで決済されないことをテスト
    （日足では高値と安値の順番が分からないため、同じ足の高値をその足の判定に使わない）
    """
    df = _create_rise_then_fall_df()
    # 高値 1100 を付けた足で、更新後のライン 1056 を割る（更新前のライン 1060 × 0.96 = 1017.6 は割らない）
    _set_bar(df, ENTRY + 2, 1090, 1100, 1050, 1090)

    trades = _build(is_trailing=True).run(df.copy()).trades.records_readable
    assert len(trades) == 1
    assert trades["Exit Timestamp"].iloc[0] == df.index[ENTRY + 3]
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(1100 * 0.96)


def test_trailing_includes_entry_day_high():
    """
    エントリー当日に SL/TP ラインへ届かない日も、当日の高値がトレーリングの最高値に入ることをテスト
    （エントリーのある日を全て2行に分けるため）
    """
    df = _create_flat_df()
    _set_bar(df, ENTRY, 1000, 1100, 990, 1080)  # エントリー当日の高値 1100（安値はライン 960 に届かない）
    _set_bar(df, ENTRY + 1, 1080, 1085, 1050, 1080)  # 1100 × 0.96 = 1056 を割る
    for i in range(ENTRY + 2, len(df)):
        _set_bar(df, i, 1080, 1090, 1070, 1080)

    portfolio = _build(is_trailing=True).run(df.copy())
    assert len(portfolio.wrapper.index) == len(df) + 1, "エントリー当日の足が分かれているはず"

    trades = portfolio.trades.records_readable
    assert len(trades) == 1
    assert trades["Exit Timestamp"].iloc[0] == df.index[ENTRY + 1]
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(1100 * 0.96)

    # トレーリングなしでは、ラインへ届かない日は分けない
    assert len(_build(is_trailing=False).run(df.copy()).wrapper.index) == len(df)


def _build_narrow_stop_backtester(is_trailing: bool) -> Backtester:
    """決済が多く出るように、SL/TP の幅を狭くした戦略"""
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(3),
        sl=AtrStopLoss(1, is_trailing=is_trailing),
        position_sizer=RiskPercentageSizer(0.02),
    )


def test_non_trailing_matches_previous_behavior():
    """
    is_trailing=False では、sl_trail を渡さない（変更前の）呼び出しと同じ結果になることをテスト
    """
    df = create_random_walk_df(days=500, seed=3)
    backtester = _build_narrow_stop_backtester(is_trailing=False)
    actual = backtester.run(df.copy())

    kwargs = backtester.prepare(df.copy()).split_entry_day_bars(backtester.slippage).to_vbt_kwargs()
    assert kwargs.pop("sl_trail") is False
    expected = vbt.Portfolio.from_signals(
        **kwargs, init_cash=backtester.init_cash, **backtester.cost_params, **JP_MARKET_PARAMS
    )
    np.testing.assert_allclose(actual.value().to_numpy(), expected.value().to_numpy())


def test_trailing_changes_trades():
    """トレーリングを有効にすると、実際に売買結果が変わることをテスト（sl_trail が渡っていることの確認）"""
    df = create_random_walk_df(days=500, seed=3)
    trailing = _build_narrow_stop_backtester(is_trailing=True).run(df.copy())
    fixed = _build_narrow_stop_backtester(is_trailing=False).run(df.copy())
    assert trailing.trades.count() > 0
    assert trailing.final_value() != pytest.approx(fixed.final_value())


def test_trailing_does_not_look_ahead():
    """
    ある日以降の価格を書き換えても、その日より前に閉じたトレードが変わらないことをテスト
    （トレーリングの最高値の更新や、エントリー日の足の分割で、未来の情報が混ざっていないことの確認）
    """
    df = create_random_walk_df(days=500, seed=3)
    cutoff = df.index[300]
    changed = df.copy()
    future = changed.index >= cutoff
    changed.loc[future, ["Open", "High", "Low", "Close"]] *= 1.3

    def _closed_before(data: pd.DataFrame) -> pd.DataFrame:
        trades = _build_narrow_stop_backtester(is_trailing=True).run(data.copy()).trades.records_readable
        closed = trades[pd.to_datetime(trades["Exit Timestamp"]) < cutoff]
        return closed[["Entry Timestamp", "Exit Timestamp", "Size", "PnL"]].reset_index(drop=True)

    expected = _closed_before(df)
    assert len(expected) > 0, "比較できるトレードがあるデータでテストする"
    pd.testing.assert_frame_equal(_closed_before(changed), expected)


def test_run_portfolio_uses_trailing_stop():
    """資金共有モードでも sl_trail が渡り、1銘柄なら Backtester.run と同じ資産推移になることをテスト"""
    df = create_random_walk_df(days=500, seed=3)

    def factory() -> Backtester:
        return _build_narrow_stop_backtester(is_trailing=True)

    expected = factory().run(df.copy())
    assert len(expected.wrapper.index) > len(df), "足を分ける日が出るデータでテストする"

    actual = MultiBacktester(factory).run_portfolio({"AAA": df})
    np.testing.assert_allclose(actual.value().to_numpy().ravel(), expected.value().to_numpy())
    # 分割した 12:00 の行は、元の日付の行の直後に入っている
    rest_index = cast(pd.DatetimeIndex, actual.wrapper.index.difference(df.index))
    assert len(rest_index) > 0
    assert (rest_index - SPLIT_BAR_OFFSET).isin(df.index).all()



def _expected_trailing_exit(
    daily: pd.DataFrame, entry_day: int, last_day: int, fill: float, sl: float, tp: float
) -> tuple[int | None, float | None]:
    """
    vectorbt とは別に、日足からトレーリング SL / TP で決済される日と、SL の決済価格（スリッページ前）を計算します。

    - SL ラインは「最高値 × (1 - エントリー日の損切り幅)」。最高値の初期値は約定価格
    - エントリー当日から、各日で SL → TP の順に判定し、その後で最高値を当日の高値で更新する
      （当日の高値は翌日から効く。エントリー当日の高値も最高値に入る）
    - 始値がラインより下で寄った場合は、始値で決済する

    Returns:
        tuple[int | None, float | None]: (決済される日の位置, SL の決済価格)。TP なら価格は None、
            last_day までに SL/TP に届かなければ (None, None)。
    """
    open_, high, low = (daily[c].to_numpy() for c in ("Open", "High", "Low"))
    peak = fill
    for j in range(entry_day, last_day + 1):
        line = peak * (1 - sl)
        if low[j] <= line:
            return j, min(open_[j], line)
        if high[j] >= fill * (1 + tp):
            return j, None
        peak = max(peak, high[j])
    return None, None


@pytest.mark.parametrize("tp_mult", [3.0, 100.0])
def test_trailing_exits_match_independent_calculation(tp_mult: float):
    """
    複数銘柄の全トレードで、トレーリング SL の決済日・決済価格が、日足から別に計算したものと一致することをテスト
    （利確なし（tp_mult=100）では、ほとんどの決済がトレーリング SL になる）。
    足の分割に頼らずに日足で計算するため、エントリー当日の高値の扱いもあわせて確かめられる
    """
    data = {f"S{seed}": create_random_walk_df(days=500, seed=seed) for seed in range(5)}

    def factory() -> Backtester:
        return Backtester(
            indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
            entries=[GoldenCrossSignal()],
            exits=[DeadCrossSignal()],
            tp=AtrTakeProfit(tp_mult),
            sl=AtrStopLoss(1, is_trailing=True),
            position_sizer=RiskPercentageSizer(0.02),
        )

    num_sl_exits = 0
    mismatches = []
    for ticker, df in data.items():
        backtester = factory()
        prepared = backtester.prepare(df.copy())  # 日足のまま（足を分ける前）のシフト済みの SL/TP 幅
        trades = backtester.run(df.copy()).trades.records_readable

        closed = trades[trades["Status"] == "Closed"]
        for entry_ts, exit_ts, entry_price, exit_price in zip(
            pd.to_datetime(closed["Entry Timestamp"]),
            pd.to_datetime(closed["Exit Timestamp"]),
            closed["Avg Entry Price"],
            closed["Avg Exit Price"],
        ):
            entry_day = cast(int, df.index.get_loc(entry_ts))
            exit_day = cast(int, df.index.get_loc(exit_ts.normalize()))
            day, sl_price = _expected_trailing_exit(
                df, entry_day, exit_day, entry_price,
                float(prepared.sl_stop.iloc[entry_day]), float(prepared.tp_stop.iloc[entry_day]),
            )
            if day is None:
                # 決済日までに SL/TP に届かないなら、エグジットシグナルで決済日の始値で約定している
                # （シグナルより前に SL を割っていないことの確認にもなる）
                expected_ts = df.index[exit_day]
                expected_price = df["Open"].iloc[exit_day] * (1 - backtester.slippage)
            else:
                # エントリー当日の決済は、分けた足の2行目（12:00）になる
                expected_ts = df.index[day] + (SPLIT_BAR_OFFSET if day == entry_day else pd.Timedelta(0))
                if sl_price is None:
                    expected_price = exit_price  # TP の決済価格は照合しない（決済日だけ見る）
                else:
                    num_sl_exits += 1
                    expected_price = sl_price * (1 - backtester.slippage)
            if exit_ts != expected_ts or not np.isclose(exit_price, expected_price):
                mismatches.append((ticker, entry_ts, exit_ts, expected_ts, exit_price, expected_price))

    assert num_sl_exits >= 20, "トレーリング SL の決済が十分に出るデータでテストする"
    assert mismatches == []
