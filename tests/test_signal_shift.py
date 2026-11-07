"""
シグナルのシフト（翌日約定）を AbstractSignal.generate() が一括して行うことと、
未来の価格がシグナルに混入しないことのテスト。
"""

import datetime

import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    AbstractSignal,
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
    InDateRangeSignal,
)

from .helpers import create_random_walk_df

# 未来の価格を書き換える日（指標の助走期間より後）
CHANGE_DAYS = [30, 60, 120, 250]


class CloseUpSignal(AbstractSignal):
    """終値が前日より上がった日に True を返す、テスト用のシグナル（シフトは既定どおり）。"""

    @property
    def required_columns(self) -> list[str]:
        return ["Close"]

    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        return df["Close"] > df["Close"].shift(1)


class LeakyCloseUpSignal(CloseUpSignal):
    """終値を使うのにシフト不要と宣言した、未来参照のあるテスト用のシグナル。"""

    shift_required = False


def _indicators() -> list:
    return [MAIndicator("EMA", 5, 20), ATRIndicator(15)]


def _all_signals() -> list[AbstractSignal]:
    """lib に実装されている全てのシグナル（新しいシグナルを追加したら、ここにも足す）。"""
    return [
        GoldenCrossSignal(),
        DeadCrossSignal(),
        ATRConditionSignal(lower=2.0, upper=2.4),
        InDateRangeSignal(datetime.datetime(2020, 3, 1), datetime.datetime(2020, 9, 30)),
    ]


def _change_future(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    k日目の始値より後に分かる情報（k日目の高値・安値・終値と、k+1日目以降の全て）を書き換えます。
    k日目の始値は、k日目に約定する時点で分かっているので変えません。
    """
    changed = df.copy()
    changed.loc[changed.index[k], ["High", "Low", "Close"]] *= 1.3
    changed.iloc[k + 1 :] *= 0.7
    return changed


def _signal_with_indicators(signal: AbstractSignal, df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    for indicator in _indicators():
        indicator.generate(df)
    return signal.generate(df)


def _uses_future(signal: AbstractSignal, df: pd.DataFrame, k: int) -> bool:
    """未来の価格を書き換えたときに、k日目までのシグナルが変わるか。"""
    original = _signal_with_indicators(signal, df)
    changed = _signal_with_indicators(signal, _change_future(df, k))
    return not original.iloc[: k + 1].equals(changed.iloc[: k + 1])


@pytest.mark.parametrize("k", CHANGE_DAYS)
@pytest.mark.parametrize("signal", _all_signals(), ids=lambda s: type(s).__name__)
def test_signal_does_not_use_future_prices(signal, k):
    """
    未来の価格を書き換えても、過去（約定日まで）のシグナルが変わらないことをテスト
    """
    df = create_random_walk_df(days=300, seed=1)
    assert not _uses_future(signal, df, k)


@pytest.mark.parametrize("k", CHANGE_DAYS)
def test_prepare_does_not_use_future_prices(k):
    """
    Backtester.prepare() の出力（シグナル・SL/TP 幅・サイズ）も、
    未来の価格を書き換えて過去が変わらないことをテスト
    """
    df = create_random_walk_df(days=300, seed=1)

    def prepare(data: pd.DataFrame):
        backtester = Backtester(
            indicators=_indicators(),
            entries=[GoldenCrossSignal(), ATRConditionSignal(lower=2.0, upper=2.4)],
            exits=[DeadCrossSignal()],
            tp=AtrTakeProfit(4),
            sl=AtrStopLoss(2),
            position_sizer=RiskPercentageSizer(0.02),
        )
        return backtester.prepare(data.copy())

    original = prepare(df)
    changed = prepare(_change_future(df, k))
    for name in ["entries", "exits", "sl_stop", "tp_stop", "size"]:
        a = getattr(original, name).iloc[: k + 1]
        b = getattr(changed, name).iloc[: k + 1]
        pd.testing.assert_series_equal(a, b, obj=name)


def test_future_check_detects_leaky_signal():
    """
    終値を使うのにシフトしないシグナルを、上の未来参照のチェックで検出できることをテスト
    （チェック自体が機能していることの確認）
    """
    df = create_random_walk_df(days=300, seed=1)
    assert any(_uses_future(LeakyCloseUpSignal(), df, k) for k in CHANGE_DAYS)
    assert not any(_uses_future(CloseUpSignal(), df, k) for k in CHANGE_DAYS)


def test_generate_shifts_by_one_day_by_default():
    """
    shift_required を宣言しないシグナルは、_calculate() の結果が1日遅れて返ることをテスト
    """
    df = create_random_walk_df(days=50, seed=2)
    signal = CloseUpSignal()
    raw = signal._calculate(df)
    result = signal.generate(df)

    assert signal.shift_required is True
    assert result.dtype == bool
    assert not result.iloc[0]
    pd.testing.assert_series_equal(result.iloc[1:], raw.iloc[:-1].set_axis(df.index[1:]))


def test_generate_does_not_shift_when_not_required():
    """
    shift_required = False のシグナルは、_calculate() の結果がそのまま返ることをテスト
    """
    df = create_random_walk_df(days=50, seed=2)
    signal = InDateRangeSignal(df.index[10].to_pydatetime(), df.index[20].to_pydatetime())
    result = signal.generate(df)

    assert result.dtype == bool
    assert result[result].index.tolist() == df.index[10:21].tolist()


def test_cross_signal_is_true_on_day_after_cross():
    """
    クロス系のシグナルは、クロスが確定した日の翌日に True になることをテスト
    """
    df = create_random_walk_df(days=10, seed=0)
    df["long_ma"] = 1000.0
    df["short_ma"] = [990.0] * 5 + [1010.0] * 5  # index=5 の終値でクロス確定

    golden = GoldenCrossSignal().generate(df)
    assert golden[golden].index.tolist() == [df.index[6]]

    df["short_ma"] = [1010.0] * 5 + [990.0] * 5
    dead = DeadCrossSignal().generate(df)
    assert dead[dead].index.tolist() == [df.index[6]]


def test_backtester_combines_shifted_and_unshifted_signals():
    """
    シフトするシグナルとしないシグナルを AND で組み合わせたとき、
    それぞれのシフトの扱いのまま結合されることをテスト
    """
    df = create_random_walk_df(days=50, seed=2)
    in_range = InDateRangeSignal(df.index[10].to_pydatetime(), df.index[30].to_pydatetime())
    close_up = CloseUpSignal()
    backtester = Backtester(
        indicators=[ATRIndicator(15)],
        entries=[close_up, in_range],
        exits=[],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )
    entries = backtester.prepare(df.copy()).entries

    expected = close_up.generate(df) & in_range.generate(df)
    pd.testing.assert_series_equal(entries, expected, check_names=False)
    # 前日に終値が上がり、当日が期間内の日だけ True（期間の初日は前日の値も使う）
    up_prev_day = (df["Close"] > df["Close"].shift(1)).shift(1, fill_value=False)
    assert entries.iloc[10] == up_prev_day.iloc[10]
    assert not entries.iloc[31:].any()
