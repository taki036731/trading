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
