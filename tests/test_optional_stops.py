# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

from lib.backtester import JP_MARKET_PARAMS, Backtester
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.position_sizing import AbstractPositionSizer, RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df

# index=10 でゴールデンクロス -> index=11 の始値 1000 でエントリー
ENTRY = 11
# index=20 でデッドクロス -> index=21 の始値で決済
EXIT = 21


class FixedSharesSizer(AbstractPositionSizer):
    """テスト用: 損切り幅を使わず、毎回 100 株を買うサイザー"""

    def __init__(self):
        super().__init__(size_type=SizeType.Amount)

    @property
    def required_columns(self) -> list[str]:
        return []

    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(100.0, index=df.index)


def _create_cross_df() -> pd.DataFrame:
    """
    値動きのない DataFrame（O=C=1000, H=1010, L=990, ATR=20）。
    index=10 でゴールデンクロス、index=20 でデッドクロス。
    毎日の安値が建値 1000 を下回るため、建値のストップ（幅 0）があれば翌日に決済される。
    """
    df = create_base_df(days=30)
    df["Open"] = 1000.0
    df["Close"] = 1000.0
    df["High"] = 1010.0
    df["Low"] = 990.0
    df.loc[:, "long_ma"] = 1000
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:20], "short_ma"] = 1010
    df.loc[df.index[20:], "short_ma"] = 990
    return df


def _build(sl=None, tp=None, sizer: AbstractPositionSizer | None = None) -> Backtester:
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=tp,
        sl=sl,
        position_sizer=sizer or FixedSharesSizer(),
        fees=0.0,
        slippage=0.0,
    )


def test_no_sl_tp_exits_only_by_signal():
    """
    sl / tp を省略すると、ストップで決済されず、デッドクロスだけで手仕舞うことをテスト
    """
    df = _create_cross_df()
    backtester = _build()

    prepared = backtester.prepare(df.copy())
    assert prepared.sl_stop.isna().all()
    assert prepared.tp_stop.isna().all()
    assert prepared.sl_trail is False

    trades = backtester.run(df.copy()).trades.records_readable
    assert len(trades) == 1
    assert trades["Entry Timestamp"].iloc[0] == df.index[ENTRY]
    assert trades["Exit Timestamp"].iloc[0] == df.index[EXIT]


def test_only_tp_omitted_keeps_sl():
    """
    tp だけを省略しても、損切りは効くことをテスト
    """
    df = _create_cross_df()
    df.loc[df.index[ENTRY + 2], "Low"] = 900  # SL ライン（1000 - 2ATR = 960）を割る

    trades = _build(sl=AtrStopLoss(2)).run(df.copy()).trades.records_readable
    assert len(trades) == 1
    assert trades["Exit Timestamp"].iloc[0] == df.index[ENTRY + 2]
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(960)


def test_leading_stop_width_is_nan():
    """
    シフトで空く先頭の行や、指標の計算前の期間の SL/TP 幅が、0 ではなく NaN（ストップなし）になることをテスト
    """
    df = _create_cross_df()
    df.loc[df.index[:5], "atr"] = np.nan

    prepared = _build(sl=AtrStopLoss(2), tp=AtrTakeProfit(4)).prepare(df.copy())
    # 先頭の行はシフトで空き、index=1..5 は前日の ATR が NaN
    assert prepared.sl_stop.iloc[:6].isna().all()
    assert prepared.tp_stop.iloc[:6].isna().all()
    assert prepared.sl_stop.iloc[6:].notna().all()


def test_zero_stop_is_breakeven_but_nan_is_none_in_vectorbt():
    """
    vectorbt の仕様の確認: sl_stop=0 は建値のストップとして発動し、NaN はストップなしになることをテスト
    """
    df = _create_cross_df()
    entries = pd.Series(False, index=df.index)
    entries.iloc[ENTRY] = True

    def _run(sl_stop: float) -> pd.DataFrame:
        pf = vbt.Portfolio.from_signals(
            close=df["Close"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            price=df["Open"],
            entries=entries,
            exits=pd.Series(False, index=df.index),
            sl_stop=sl_stop,
            size=100.0,
            **JP_MARKET_PARAMS,
        )
        return pf.trades.records_readable

    zero = _run(0.0)
    assert zero["Status"].iloc[0] == "Closed"
    assert zero["Exit Timestamp"].iloc[0] == df.index[ENTRY + 1]
    assert zero["Avg Exit Price"].iloc[0] == pytest.approx(1000)

    none = _run(np.nan)
    assert none["Status"].iloc[0] == "Open"


def test_split_entry_day_bars_treats_nan_as_no_stop():
    """
    split_entry_day_bars が、NaN の幅を「ストップなし」として扱い、足を分けないことをテスト。
    あわせて、0 は建値のストップ（あり）として分けることを確認する
    """
    df = _create_cross_df()
    prepared = _build().prepare(df.copy())
    # NaN（ストップなし）: 安値 990 が建値を割っても分けない
    assert prepared.split_entry_day_bars(slippage=0.0) is prepared

    prepared.sl_stop = pd.Series(0.0, index=df.index)
    split = prepared.split_entry_day_bars(slippage=0.0)
    assert len(split.close) == len(df) + 1


def test_risk_sizer_requires_sl():
    """
    損切り幅を使う RiskPercentageSizer と sl=None を組み合わせると、例外になることをテスト
    """
    with pytest.raises(ValueError, match="sl を省略できません"):
        _build(sizer=RiskPercentageSizer(0.02))


def test_risk_sizer_returns_nan_without_stop_width():
    """
    RiskPercentageSizer が、損切り幅が NaN / 0 の日に inf ではなく NaN（注文なし）を返すことをテスト
    """
    df = _create_cross_df()
    sl_pct = pd.Series([np.nan, 0.0, 0.04], index=df.index[:3])
    size = RiskPercentageSizer(0.02).generate(df.iloc[:3], sl_pct=sl_pct)
    assert np.isnan(size.iloc[0])
    assert np.isnan(size.iloc[1])
    assert size.iloc[2] == pytest.approx(0.5)


def test_no_entry_when_stop_width_is_nan():
    """
    SL の幅が NaN の日のシグナルでは、RiskPercentageSizer で注文が出ず、建値で決済されることもないことをテスト
    """
    df = _create_cross_df()
    df.loc[df.index[:15], "atr"] = np.nan  # エントリー前日（index=10）の ATR がない

    pf = _build(sl=AtrStopLoss(2), sizer=RiskPercentageSizer(0.02)).run(df.copy())
    assert len(pf.trades.records) == 0


def test_combine_keeps_nan_stop_width():
    """
    資金共有モードでも、SL/TP の幅の NaN が 0 で埋められず、ストップなしのまま渡されることをテスト
    """
    df = _create_cross_df()
    late = df.iloc[5:].copy()  # 先頭 5 日はデータがない銘柄

    prepared = {
        "AAA": _build().prepare(df.copy()),
        "BBB": _build(sl=AtrStopLoss(2)).prepare(late),
    }
    kwargs = MultiBacktester._combine(prepared)
    assert kwargs["sl_stop"]["AAA"].isna().all()
    assert kwargs["tp_stop"].isna().all().all()
    assert kwargs["sl_stop"]["BBB"].iloc[:6].isna().all()  # データのない 5 日 + シフトの 1 日
    assert kwargs["sl_stop"]["BBB"].iloc[6:].notna().all()


def test_run_portfolio_without_stops_matches_backtester():
    """
    sl / tp を省略した戦略でも、資金共有モードの1銘柄が Backtester.run と同じ結果になることをテスト
    """
    df = _create_cross_df()
    expected = _build().run(df.copy())
    actual = MultiBacktester(_build).run_portfolio({"AAA": df})
    np.testing.assert_allclose(actual.value().values, expected.value().values)
    assert len(actual.trades.records) == 1
