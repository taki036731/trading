import numpy as np
import pandas as pd

from lib.backtester import UNIT_SHARES, Backtester
from lib.multi_backtester import MultiBacktester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import DeadCrossSignal, GoldenCrossSignal

from .helpers import create_base_df


def build_strategy(init_cash: float = 1000000.0) -> Backtester:
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
    )


def cross_df(raw_ratio: float | None, cross_day: int = 10, dead_day: int | None = None):
    """
    cross_day 日目にゴールデンクロス（翌日エントリー）、dead_day 日目にデッドクロス（翌日エグジット）が起きるデータ。
    raw_ratio を指定すると、実際の始値 RawOpen = 調整後の始値 × raw_ratio の列を加える
    """
    df = create_base_df(days=30)
    df.loc[df.index[cross_day:], "short_ma"] = 1010
    if dead_day is not None:
        df.loc[df.index[dead_day:], "short_ma"] = 990
    if raw_ratio is not None:
        df["RawOpen"] = df["Open"] * raw_ratio
    return df


def test_lot_size_is_100_without_raw_open():
    """RawOpen がなければ、単元は今までどおり100株"""
    df = create_base_df(days=5)

    lot = Backtester._lot_size(df)

    assert (lot == UNIT_SHARES).all()


def test_lot_size_converts_real_unit_to_adjusted_shares():
    """1:5 分割前（実際の株価が調整後の5倍）の100株は、調整後の500株になる"""
    df = create_base_df(days=5)
    df["RawOpen"] = df["Open"] * 5

    lot = Backtester._lot_size(df)

    np.testing.assert_allclose(lot.values, 500.0)


def test_entry_is_rounded_to_real_unit():
    """
    資金200万円の約50%（約100万円）で、エントリー価格は約1022円（調整後）。
    100株単元なら900株だが、実際の株価が5倍なら1単元=調整後500株なので500株になる
    """
    adjusted_only = build_strategy(2000000.0).run(cross_df(raw_ratio=None))
    with_raw = build_strategy(2000000.0).run(cross_df(raw_ratio=5))

    assert adjusted_only.trades.records["size"][0] == 900
    assert with_raw.trades.records["size"][0] == 500


def test_entry_is_skipped_when_real_unit_exceeds_budget():
    """
    資金100万円の約50%（約50万円）では、実際の1単元（約1022円×5×100株=約51万円）に届かず発注されない。
    調整後価格だけで判定すると、400株買えてしまう
    """
    adjusted_only = build_strategy().run(cross_df(raw_ratio=None))
    with_raw = build_strategy().run(cross_df(raw_ratio=5))

    assert adjusted_only.trades.records["size"][0] == 400
    assert len(with_raw.trades.records) == 0


def test_exit_sells_all_shares_after_lot_changes():
    """保有中に配当落ちで単元（調整後の株数）が変わっても、エグジットで全量を売る"""
    df = cross_df(raw_ratio=5, dead_day=20)
    # 15日目以降、配当落ちで実際の株価と調整後の株価の比率が変わったものとする
    df.loc[df.index[15:], "RawOpen"] = df["Open"] * 5.1

    portfolio = build_strategy(2000000.0).run(df)

    trades = portfolio.trades.records
    assert len(trades) == 1
    assert trades.status[0] == 1  # Closed
    assert portfolio.assets().iloc[-1] == 0


def test_run_portfolio_uses_real_unit():
    """資金共有モードでも、単一銘柄と同じく実際の単元で判定する"""
    df = cross_df(raw_ratio=5, dead_day=20)
    df.loc[df.index[15:], "RawOpen"] = df["Open"] * 5.1

    expected = build_strategy(2000000.0).run(df.copy())
    actual = MultiBacktester(lambda: build_strategy(2000000.0)).run_portfolio(
        {"AAA": df}
    )

    np.testing.assert_allclose(actual.value().values, expected.value().values)
    assert actual.trades.records["size"][0] == 500
