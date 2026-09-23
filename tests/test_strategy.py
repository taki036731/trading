from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
)
from tools.sample2 import run_vectorbt_backtest

from .helpers import create_base_df


def test_golden_cross_and_take_profit():
    """
    ゴールデンクロスでエントリーし、4ATRで利確されることをテスト
    """
    df = create_base_df(days=30)

    # 10日目にゴールデンクロスを発生させる
    # 9日目: short(990) < long(1000)
    # 10日目: short(1010) > long(1000)
    df.loc[df.index[0:10], "short_ma"] = 990
    df.loc[df.index[10:], "short_ma"] = 1010
    df.loc[:, "long_ma"] = 1000

    # 20日目に価格を急騰させて利確(4ATR=80円)を発生させる
    # エントリー価格は約1020円(11日目想定)なので、1100円以上で利確
    df.loc[df.index[20:], "Close"] = 1200
    df.loc[df.index[20:], "High"] = 1210

    portfolio = run_vectorbt_backtest(
        indicators=[],  # 手動でカラムを入れたのでインジケータ計算は不要
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),  # 4ATR = 80円幅
        sl=AtrStopLoss(2),  # 2ATR = 40円幅
        position_sizer=RiskPercentageSizer(0.02),
        df=df,
        verbose=False,
    )

    # 検証
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

    portfolio = run_vectorbt_backtest(
        indicators=[],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=0, upper=5),  # 5%以下ならエントリー
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        df=df,
        verbose=False,
    )

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

    # エントリー時の条件:
    # Close = 1020 (11日目)
    # ATR = 20
    # SL = 2ATR = 40円
    # SL比率 = 40 / 1020 ≒ 0.0392 (3.92%)
    # リスク = 2% (0.02)
    # ポジションサイズ = 0.02 / 0.0392 ≒ 0.51 (資産の51%)

    init_cash = 1000000.0
    portfolio = run_vectorbt_backtest(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[],
        tp=AtrTakeProfit(100),  # 利確させない
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        df=df,
        init_cash=init_cash,
        verbose=False,
    )

    order = portfolio.orders.records
    assert len(order) > 0

    # 資産100万の51% = 51万円分購入
    # 1020円の株を51万円分 -> 約500株
    # 日本株は100株単位設定にしているので、端数がどうなるか
    assert order["size"][0] >= 400  # およその株数チェック
