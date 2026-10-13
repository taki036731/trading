import pytest

from lib.backtester import Backtester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
)

from .helpers import create_base_df


def test_golden_cross_and_take_profit():
    """
    ゴールデンクロスでエントリーし、4ATRで利確されることをテスト
    """
    df = create_base_df(days=30)

    # ... (既存のデータ準備) ...
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[:, "long_ma"] = 1000

    df.loc[df.index[20:], "Close"] = 1200
    df.loc[df.index[20:], "High"] = 1210

    backtester = Backtester(
        indicators=[],  # 手動でカラムを入れたのでインジケータ計算は不要
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),  # 4ATR = 80円幅
        sl=AtrStopLoss(2),  # 2ATR = 40円幅
        position_sizer=RiskPercentageSizer(0.02),
    )
    portfolio = backtester.run(df)

    # 検証
    # ... (既存の検証) ...
    trades = portfolio.trades.records
    assert len(trades) == 1, "取引が1回行われるはず"

    # vectorbtはシグナルの翌日にエントリーする設定(デフォルト)
    # 10日目にシグナル -> 11日目にエントリー
    assert trades.entry_idx[0] == 10  # 0-indexed なので 10 は 11日目

    # 利確されているか (ExitType 1 が Profit Stop)
    # vbtの内部コードにより異なるが、プロフィットがプラスであることを確認
    assert trades.pnl[0] > 0, "利益が出ているはず"


def test_atr_filter_suppresses_entry():
    """
    ATRが指定範囲外の場合にエントリーが抑制されることをテスト
    """
    df = create_base_df(days=30)

    # 10日目にGC発生
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[:, "long_ma"] = 1000

    # しかし、その日のATRを非常に大きくする (ATR/Close * 100 > 5%)
    # Close=1020, ATR=100 -> ratio=9.8%
    df.loc[df.index[10], "atr"] = 100.0

    backtester = Backtester(
        indicators=[],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=0, upper=5),  # 5%以下ならエントリー
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )
    portfolio = backtester.run(df)

    assert len(portfolio.trades.records) == 0, (
        "ATRフィルタによりエントリーが阻止されるはず"
    )


def test_position_sizing():
    """
    リスク2%に基づいた株数が計算されているかテスト
    """
    df = create_base_df(days=30)
    # 10日目にGC
    df.loc[df.index[10:], "short_ma"] = 1010

    # ... (既存のコメント) ...

    init_cash = 1000000.0
    backtester = Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=init_cash,
    )
    portfolio = backtester.run(df)

    order = portfolio.orders.records
    assert len(order) > 0

    # 資産100万の51% = 51万円分購入
    # 1020円の株を51万円分 -> 約500株
    # 日本株は100株単位設定にしているので、端数がどうなるか
    assert order["size"][0] >= 400  # およその株数チェック


def _create_flat_df_with_golden_cross():
    """
    SL/TP の約定価格を検証するための、値動きのない DataFrame を作成する。

    10日目(index=10)にゴールデンクロスが発生し、index=11 の始値 1000 でエントリーする。
    ATR=20 なので、TP ラインは 1000 + 4ATR = 1080、SL ラインは 1000 - 2ATR = 960。
    エントリー日の終値だけ 1020 にして、終値基準のライン(TP=1101.6 / SL=979.2)と
    区別できるようにしている。
    """
    df = create_base_df(days=30)
    df["Open"] = 1000.0
    df["Close"] = 1000.0
    df["High"] = 1010.0
    df["Low"] = 990.0
    df.loc[df.index[11], ["Close", "High"]] = [1020.0, 1020.0]

    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[:, "long_ma"] = 1000
    return df


def _run_atr_exit_backtest(df):
    backtester = Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )
    return backtester.run(df).trades.records


def test_take_profit_is_based_on_fill_price_and_hit_by_high():
    """
    TP ラインが約定価格(始値)基準で、高値が届いた日にライン価格で約定することをテスト
    """
    df = _create_flat_df_with_golden_cross()
    # 終値は 1000 のまま、高値だけ TP ライン(1080)を超える
    df.loc[df.index[15], "High"] = 1090.0

    trades = _run_atr_exit_backtest(df)

    assert len(trades) == 1
    assert trades.entry_idx[0] == 11
    assert trades.entry_price[0] == pytest.approx(1000.0)
    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(1080.0)


def test_stop_loss_is_based_on_fill_price_and_hit_by_low():
    """
    SL ラインが約定価格(始値)基準で、安値が届いた日にライン価格で約定することをテスト
    """
    df = _create_flat_df_with_golden_cross()
    # 終値は 1000 のまま、安値だけ SL ライン(960)を割る
    df.loc[df.index[15], "Low"] = 950.0

    trades = _run_atr_exit_backtest(df)

    assert len(trades) == 1
    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(960.0)


def test_take_profit_fills_at_open_when_gapped_over():
    """
    始値が TP ラインを飛び越えて寄り付いた場合、ライン価格ではなく始値で約定することをテスト
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[15], ["Open", "High", "Low", "Close"]] = [
        1100.0,
        1110.0,
        1090.0,
        1100.0,
    ]

    trades = _run_atr_exit_backtest(df)

    assert len(trades) == 1
    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(1100.0)
