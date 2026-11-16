# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import logging

import numpy as np
import pandas as pd
import pytest
from vectorbt.portfolio.enums import SizeType

from lib.backtester import Backtester
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss
from lib.strategies.position_sizing import AbstractPositionSizer, RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df

# index=10 でゴールデンクロス -> index=11 の始値でエントリー
ENTRY = 11


class ConstantSizer(AbstractPositionSizer):
    """テスト用: 毎日同じサイズを返すサイザー"""

    def __init__(self, value: float, max_size: float | None = None):
        super().__init__(size_type=SizeType.Amount, max_size=max_size)
        self.value = value

    @property
    def required_columns(self) -> list[str]:
        return []

    def _calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(self.value, index=df.index)


def _create_cross_df(atr: float) -> pd.DataFrame:
    """
    値動きのない DataFrame（O=C=1000, H=1010, L=990）。index=10 でゴールデンクロス、index=20 でデッドクロス。
    損切り幅は 2ATR / 1000 なので、atr=5 なら 1% になり、リスク 2% のサイズは 2.0（200%）になる
    """
    df = create_base_df(days=30)
    df["Open"] = 1000.0
    df["Close"] = 1000.0
    df["High"] = 1010.0
    df["Low"] = 990.0
    df["atr"] = atr
    df.loc[:, "long_ma"] = 1000
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:20], "short_ma"] = 1010
    df.loc[df.index[20:], "short_ma"] = 990
    return df


def _build(max_size: float | None = 1.0, init_cash: float = 1_000_000.0) -> Backtester:
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02, max_size=max_size),
        init_cash=init_cash,
        fees=0.0,
        slippage=0.0,
    )


def test_risk_sizer_clips_at_max_size():
    """
    RiskPercentageSizer が、既定値で 1.0 を超えるサイズを 1.0 で切り、NaN はそのまま残すことをテスト
    """
    df = _create_cross_df(atr=5)
    sl_pct = pd.Series([np.nan, 0.01, 0.04, 0.02], index=df.index[:4])
    size = RiskPercentageSizer(0.02).generate(df.iloc[:4], sl_pct=sl_pct)
    assert np.isnan(size.iloc[0])
    assert size.iloc[1] == pytest.approx(1.0)  # 2.0 -> 1.0
    assert size.iloc[2] == pytest.approx(0.5)
    assert size.iloc[3] == pytest.approx(1.0)


def test_risk_sizer_custom_and_no_max_size():
    """
    max_size を変えるとその値で切られ、None なら上限がかからないことをテスト
    """
    df = _create_cross_df(atr=5)
    sl_pct = pd.Series(0.01, index=df.index[:2])
    assert RiskPercentageSizer(0.02, max_size=0.5).generate(df.iloc[:2], sl_pct=sl_pct).iloc[0] == pytest.approx(0.5)
    assert RiskPercentageSizer(0.02, max_size=None).generate(df.iloc[:2], sl_pct=sl_pct).iloc[0] == pytest.approx(2.0)


def test_base_sizer_has_no_max_size_by_default():
    """
    基底クラスの max_size は既定値で None（上限なし）で、指定すれば他のサイザーでも上限が効くことをテスト
    """
    df = _create_cross_df(atr=20)
    assert ConstantSizer(500.0).max_size is None
    assert (ConstantSizer(500.0).generate(df) == 500.0).all()
    assert (ConstantSizer(500.0, max_size=300.0).generate(df) == 300.0).all()


@pytest.mark.parametrize("max_size", [0.0, -1.0])
def test_max_size_must_be_positive(max_size: float):
    """
    max_size が 0 以下だと例外になることをテスト
    """
    with pytest.raises(ValueError, match="max_size"):
        RiskPercentageSizer(0.02, max_size=max_size)


def test_warns_capped_entries(caplog: pytest.LogCaptureFixture):
    """
    サイズが上限に達したエントリーがあると、エントリーの日だけを数えて警告することをテスト。
    （atr=5 では毎日サイズが上限を超えるが、エントリーは1回だけ）
    """
    df = _create_cross_df(atr=5)
    with caplog.at_level(logging.WARNING, logger="lib.backtester"):
        _build().run(df.copy(), name="AAA")
    messages = [r.getMessage() for r in caplog.records if "上限" in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].startswith("[AAA] ")
    assert "1/1 件" in messages[0]


def test_no_warning_below_max_size(caplog: pytest.LogCaptureFixture):
    """
    サイズが上限に届かない場合（atr=20 で損切り幅 4%、サイズ 0.5）や、上限なしの場合は警告しないことをテスト
    """
    with caplog.at_level(logging.WARNING, logger="lib.backtester"):
        _build().prepare(_create_cross_df(atr=20))
        _build(max_size=None).prepare(_create_cross_df(atr=5))
    assert not [r for r in caplog.records if "上限" in r.getMessage()]


def test_clipping_does_not_change_vectorbt_result():
    """
    1.0 で上限を切っても、切らない場合（vectorbt が現金全額に丸める）と約定が同じになることをテスト
    """
    df = _create_cross_df(atr=5)
    clipped = _build(max_size=1.0).run(df.copy())
    raw = _build(max_size=None).run(df.copy())
    assert len(clipped.trades.records) == 1
    np.testing.assert_allclose(clipped.value().values, raw.value().values)
    np.testing.assert_allclose(
        clipped.trades.records["size"], raw.trades.records["size"]
    )


def test_capped_entries_are_first_come_in_data_order(caplog: pytest.LogCaptureFixture):
    """
    資金共有モード（call_seq="auto"）では、同じ日の買いがサイズの小さい順に処理されるため、
    上限に達した銘柄同士は同じサイズ（1.0）になり、data の並び順で先着になることをテスト。
    あわせて、銘柄名付きで警告することを確認する。

    AAA は損切り幅 1%（サイズ 2.0）、BBB は 1.5%（サイズ 1.33）。
    上限を切らないと、サイズの小さい BBB が先に現金を全額使う
    """
    data = {"AAA": _create_cross_df(atr=5), "BBB": _create_cross_df(atr=7.5)}

    with caplog.at_level(logging.WARNING, logger="lib.backtester"):
        clipped = MultiBacktester(lambda: _build(max_size=1.0)).run_portfolio(data)
    raw = MultiBacktester(lambda: _build(max_size=None)).run_portfolio(data)

    assert clipped.trades.records_readable["Column"].tolist() == ["AAA"]
    assert raw.trades.records_readable["Column"].tolist() == ["BBB"]

    messages = [r.getMessage() for r in caplog.records if "上限" in r.getMessage()]
    assert any(m.startswith("[資金共有 {} AAA] ") for m in messages)
    assert any(m.startswith("[資金共有 {} BBB] ") for m in messages)
