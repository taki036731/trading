# vectorbt の portfolio.trades などを Pylance が関数と見なすため、このファイルではこのルールを無視する
# pyright: reportFunctionMemberAccess=false

import pandas as pd
import pytest
import vectorbt as vbt

from lib.backtester import (
    DEFAULT_SLIPPAGE,
    JP_MARKET_PARAMS,
    SPLIT_BAR_OFFSET,
    Backtester,
)
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
)

from .helpers import create_base_df, create_random_walk_df


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

    # クロス系シグナルは shift(1) されるため、シグナルの翌日にエントリーする
    # index=10 の終値でクロス確定 -> index=11 の始値でエントリー
    assert trades.entry_idx[0] == 11

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


def _build_atr_exit_backtester(**costs) -> Backtester:
    # ラインの計算を検証しやすいように、指定がなければ取引コストなしで実行する
    costs = {"fees": 0.0, "slippage": 0.0, **costs}
    return Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        **costs,
    )


def _run_atr_exit_backtest(df, **costs):
    return _build_atr_exit_backtester(**costs).run(df).trades.records


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


def test_default_slippage_is_applied():
    """
    引数を省略した場合、既定のスリッページで約定することをテスト
    """
    df = _create_flat_df_with_golden_cross()
    backtester = Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )
    trades = backtester.run(df).trades.records

    assert DEFAULT_SLIPPAGE > 0
    assert trades.entry_price[0] == pytest.approx(1000.0 * (1 + DEFAULT_SLIPPAGE))


def test_slippage_on_signal_exit():
    """
    シグナルでの売買が、買いは高く、売りは安く約定することをテスト
    """
    df = _create_flat_df_with_golden_cross()
    # index=15 の終値でデッドクロス確定 -> index=16 の始値でエグジット
    df.loc[df.index[15:], "short_ma"] = 990

    trades = _run_atr_exit_backtest(df, slippage=0.001)

    assert len(trades) == 1
    assert trades.entry_price[0] == pytest.approx(1001.0)
    assert trades.exit_idx[0] == 16
    assert trades.exit_price[0] == pytest.approx(999.0)


def test_slippage_on_take_profit():
    """
    TP の決済にもスリッページがかかることをテスト（stop_exit_price の既定値では0になる）
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[15], "High"] = 1090.0

    trades = _run_atr_exit_backtest(df, slippage=0.001)

    # ラインはスリッページ込みの約定価格が基準: 1001 * 1.08 = 1081.08
    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(1001.0 * 1.08 * 0.999)


def test_slippage_on_stop_loss():
    """
    SL の決済にもスリッページがかかることをテスト（stop_exit_price の既定値では0になる）
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[15], "Low"] = 950.0

    trades = _run_atr_exit_backtest(df, slippage=0.001)

    # ラインはスリッページ込みの約定価格が基準: 1001 * 0.96 = 960.96
    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(1001.0 * 0.96 * 0.999)


def test_slippage_on_stop_loss_gapped_down():
    """
    始値が SL ラインを割って寄り付いた場合、始値を基準にスリッページがかかることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[15], ["Open", "High", "Low", "Close"]] = [
        900.0,
        910.0,
        890.0,
        900.0,
    ]

    trades = _run_atr_exit_backtest(df, slippage=0.001)

    assert trades.exit_idx[0] == 15
    assert trades.exit_price[0] == pytest.approx(900.0 * 0.999)


def test_fees_are_charged_on_both_sides():
    """
    手数料が、買いと売りの両方の売買代金にかかることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[15], "High"] = 1090.0

    trades = _run_atr_exit_backtest(df, fees=0.001)

    size = trades["size"][0]
    assert size > 0
    assert trades.entry_fees[0] == pytest.approx(size * 1000.0 * 0.001)
    assert trades.exit_fees[0] == pytest.approx(size * 1080.0 * 0.001)


# ---------------------------------------------------------
# エントリー当日の SL/TP（足を2行に分けて判定する）
# ---------------------------------------------------------


def _run_entry_day_backtest(df, **costs) -> pd.DataFrame:
    """index=11 の始値でエントリーするデータを実行し、トレードの一覧（読みやすい形式）を返す"""
    portfolio = _build_atr_exit_backtester(**costs).run(df)
    assert portfolio.assets().iloc[-1] == 0, "全量が決済されているはず"
    return portfolio.trades.records_readable


def test_stop_loss_on_entry_day():
    """
    エントリー当日の安値が SL ラインを割った場合、その日のうちにライン価格で決済されることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[11], "Low"] = 950.0  # SL ライン 960 を割る

    trades = _run_entry_day_backtest(df)

    assert len(trades) == 1
    assert trades["Entry Timestamp"].iloc[0] == df.index[11]
    assert trades["Avg Entry Price"].iloc[0] == pytest.approx(1000.0)
    # 決済は同じ日の2行目（始値より後の値動き）
    assert trades["Exit Timestamp"].iloc[0] == df.index[11] + SPLIT_BAR_OFFSET
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(960.0)


def test_take_profit_on_entry_day():
    """
    エントリー当日の高値が TP ラインに届いた場合、その日のうちにライン価格で決済されることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[11], "High"] = 1090.0  # TP ライン 1080 に届く

    trades = _run_entry_day_backtest(df)

    assert len(trades) == 1
    assert trades["Exit Timestamp"].iloc[0] == df.index[11] + SPLIT_BAR_OFFSET
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(1080.0)


def test_stop_loss_wins_when_both_hit_on_entry_day():
    """
    エントリー当日に SL と TP の両方に届いた場合、SL が優先されることをテスト（保守的な側）
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[11], ["High", "Low"]] = [1090.0, 950.0]

    trades = _run_entry_day_backtest(df)

    assert len(trades) == 1
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(960.0)


def test_entry_day_stop_line_is_based_on_fill_price_with_slippage():
    """
    スリッページがある場合、当日の判定も約定価格基準のラインで行うことをテスト

    始値 1000 で約定価格は 1001。ラインは 1001 * 0.96 = 960.96 で、始値基準のライン 960 より高い。
    安値をその間に置き、始値基準の判定では漏れるケースでも当日に決済されることを確認する。
    """
    df = _create_flat_df_with_golden_cross()
    df.loc[df.index[11], "Low"] = 960.5

    trades = _run_entry_day_backtest(df, slippage=0.001)

    assert len(trades) == 1
    assert trades["Exit Timestamp"].iloc[0] == df.index[11] + SPLIT_BAR_OFFSET
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(1001.0 * 0.96 * 0.999)


def test_no_split_when_lines_not_touched_on_entry_day():
    """
    エントリー当日にラインへ届かない場合は足を分けず、Portfolio のインデックスも元のままであることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    backtester = _build_atr_exit_backtester()

    prepared = backtester.prepare(df.copy())
    assert prepared.split_entry_day_bars(backtester.slippage) is prepared

    portfolio = backtester.run(df)
    assert portfolio.wrapper.index.equals(df.index)


def test_existing_position_is_stopped_on_split_day():
    """
    保有中にエントリーシグナルが出て足を分けた日も、既存の建玉の SL が判定され、全量が決済されることをテスト
    """
    df = _create_flat_df_with_golden_cross()
    # index=10 と index=14 にゴールデンクロス -> index=11 でエントリーし、index=15 のシグナルは保有中
    df.loc[df.index[13], "short_ma"] = 990
    df.loc[df.index[15], "Low"] = 955.0  # index=15 のシグナルで足を分け、既存の建玉の SL ライン 960 も割る

    backtester = Backtester(
        indicators=[],
        entries=[GoldenCrossSignal()],
        exits=[],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        fees=0.0,
        slippage=0.0,
    )
    portfolio = backtester.run(df)
    trades = portfolio.trades.records_readable

    assert len(portfolio.wrapper.index) == len(df) + 1, "index=15 の足が分かれているはず"
    assert len(trades) == 1
    assert trades["Entry Timestamp"].iloc[0] == df.index[11]
    assert trades["Exit Timestamp"].iloc[0] == df.index[15] + SPLIT_BAR_OFFSET
    assert trades["Avg Exit Price"].iloc[0] == pytest.approx(960.0)
    assert portfolio.assets().iloc[-1] == 0


def _build_narrow_stop_backtester() -> Backtester:
    """エントリー当日にラインへ届く日が多く出るように、SL/TP の幅を狭くした戦略"""
    return Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        entries=[GoldenCrossSignal()],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(1),
        sl=AtrStopLoss(0.5),
        position_sizer=RiskPercentageSizer(0.02),
    )


def test_split_keeps_shifted_inputs_on_original_rows():
    """
    足を分けても、元の日付の行のシグナル・SL/TP 幅・サイズ（shift(1) 済み）が変わらないことをテスト
    """
    df = create_random_walk_df(days=500, seed=3)
    backtester = _build_narrow_stop_backtester()
    prepared = backtester.prepare(df.copy())
    split = prepared.split_entry_day_bars(backtester.slippage)

    original_index = prepared.open.index
    rest_index = split.open.index.difference(original_index)
    assert len(rest_index) > 0, "足を分ける日が出るデータでテストする"
    assert split.open.index.dtype == original_index.dtype

    for attr in ["entries", "exits", "sl_stop", "tp_stop", "size", "size_granularity", "min_size", "price", "open"]:
        # 分割後のインデックスは freq（営業日など）を持たないため、値だけを比べる
        pd.testing.assert_series_equal(
            getattr(split, attr).loc[original_index],
            getattr(prepared, attr),
            check_names=False,
            check_freq=False,
        )

    # 2行目にはシグナルを出さない
    assert not split.entries.loc[rest_index].any()
    assert not split.exits.loc[rest_index].any()

    # 2行目は元の高値・安値・終値、1行目は始値だけの足
    days = rest_index - SPLIT_BAR_OFFSET
    for attr in ["high", "low", "close"]:
        assert (getattr(split, attr).loc[rest_index].to_numpy() == getattr(prepared, attr).loc[days].to_numpy()).all()
        assert (getattr(split, attr).loc[days] == prepared.open.loc[days]).all()


def test_split_does_not_change_trades_before_first_split_day():
    """
    最初に足を分けた日より前に閉じたトレードは、分けない場合と完全に一致することをテスト
    （分割で未来の情報が過去の売買に混ざっていないことの確認）
    """
    df = create_random_walk_df(days=500, seed=3)
    backtester = _build_narrow_stop_backtester()
    prepared = backtester.prepare(df.copy())
    split = prepared.split_entry_day_bars(backtester.slippage)
    first_split_day = split.open.index.difference(prepared.open.index)[0] - SPLIT_BAR_OFFSET

    def _closed_before(kwargs: dict) -> pd.DataFrame:
        portfolio = vbt.Portfolio.from_signals(
            **kwargs, init_cash=backtester.init_cash, **backtester.cost_params, **JP_MARKET_PARAMS
        )
        trades = portfolio.trades.records_readable
        closed = trades[pd.to_datetime(trades["Exit Timestamp"]) < first_split_day]
        return closed[["Entry Timestamp", "Exit Timestamp", "Size", "PnL"]].reset_index(drop=True)

    expected = _closed_before(prepared.to_vbt_kwargs())
    actual = _closed_before(split.to_vbt_kwargs())

    assert len(expected) > 0, "比較できるトレードがあるデータでテストする"
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
