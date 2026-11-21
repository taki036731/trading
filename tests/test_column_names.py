# pyright: reportFunctionMemberAccess=false
import pandas as pd
import pytest

from lib.backtester import Backtester
from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import ATRConditionSignal, DeadCrossSignal, GoldenCrossSignal
from tests.helpers import create_random_walk_df


def build_strategy(**overrides) -> Backtester:
    """既定の列名で組んだ戦略。overrides で部品を差し替える"""
    parts = {
        "indicators": [MAIndicator("EMA", 5, 20), ATRIndicator(14)],
        "entries": [GoldenCrossSignal(), ATRConditionSignal(lower=0.5, upper=10)],
        "exits": [DeadCrossSignal()],
        "tp": AtrTakeProfit(4),
        "sl": AtrStopLoss(2),
        "position_sizer": RiskPercentageSizer(0.02),
    }
    parts.update(overrides)
    return Backtester(**parts)


def test_indicators_write_to_given_columns():
    """indicator は引数で指定した列名に出力する"""
    df = create_random_walk_df()

    MAIndicator("EMA", 5, 20, short_col="ema5", long_col="ema20").generate(df)
    ATRIndicator(14, col="atr14").generate(df)

    assert {"ema5", "ema20", "atr14"} <= set(df.columns)
    assert not {"short_ma", "long_ma", "atr"} & set(df.columns)


def test_same_indicator_twice_with_different_columns():
    """同じ種類の indicator を別の列名で2つ使える（例: 損切り用の ATR14、フィルター用の ATR50）"""
    df = create_random_walk_df()

    ATRIndicator(14, col="atr14").generate(df)
    ATRIndicator(50, col="atr50").generate(df)

    assert not df["atr14"].equals(df["atr50"])
    assert df["atr50"].isna().sum() > df["atr14"].isna().sum()


def test_indicator_raises_instead_of_overwriting():
    """既にある列に出力しようとしたら、上書きせずに例外にする"""
    df = create_random_walk_df()
    ATRIndicator(14).generate(df)
    before = df["atr"].copy()

    with pytest.raises(ValueError, match="'atr'"):
        ATRIndicator(50).generate(df)
    pd.testing.assert_series_equal(df["atr"], before)


def test_indicator_raises_on_price_column():
    """価格の列（Close など）を出力先にしたら例外にする"""
    with pytest.raises(ValueError, match="'Close'"):
        ATRIndicator(14, col="Close").generate(create_random_walk_df())


def test_atr_indicator_requires_high_and_low():
    """ATR の計算に必要な High / Low の欠落を検出する"""
    df = create_random_walk_df().drop(columns=["High"])

    with pytest.raises(ValueError, match="'High'"):
        ATRIndicator(14).generate(df)


def test_backtester_rejects_duplicate_output_columns():
    """同じ列を出力する indicator を2つ渡したら、データを読む前に例外にする"""
    with pytest.raises(ValueError, match="'atr'"):
        build_strategy(indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(14), ATRIndicator(50)])


def test_components_read_given_columns():
    """signal / exit は引数で指定した列名を読む。既定の列名と同じ結果になる"""
    df = create_random_walk_df()
    MAIndicator("EMA", 5, 20).generate(df)
    ATRIndicator(14).generate(df)
    renamed = df.rename(columns={"short_ma": "s", "long_ma": "l", "atr": "a"})

    pairs = [
        (GoldenCrossSignal(), GoldenCrossSignal(short_col="s", long_col="l")),
        (DeadCrossSignal(), DeadCrossSignal(short_col="s", long_col="l")),
        (ATRConditionSignal(0.5, 3), ATRConditionSignal(0.5, 3, atr_col="a")),
        (AtrStopLoss(2), AtrStopLoss(2, atr_col="a")),
        (AtrTakeProfit(4), AtrTakeProfit(4, atr_col="a")),
    ]
    for default, custom in pairs:
        pd.testing.assert_series_equal(default.generate(df), custom.generate(renamed))


def test_custom_column_names_give_same_backtest():
    """列名を変えても、同じ指標なら同じ売買結果になる"""
    df = create_random_walk_df(seed=1)
    custom = build_strategy(
        indicators=[MAIndicator("EMA", 5, 20, short_col="s", long_col="l"), ATRIndicator(14, col="a")],
        entries=[
            GoldenCrossSignal(short_col="s", long_col="l"),
            ATRConditionSignal(lower=0.5, upper=10, atr_col="a"),
        ],
        exits=[DeadCrossSignal(short_col="s", long_col="l")],
        tp=AtrTakeProfit(4, atr_col="a"),
        sl=AtrStopLoss(2, atr_col="a"),
    )

    default_pf = build_strategy().run(df)
    custom_pf = custom.run(df)

    assert len(default_pf.trades.records) > 0, "取引が発生するデータでテストする"
    pd.testing.assert_series_equal(default_pf.value(), custom_pf.value())


def test_run_does_not_modify_input_and_can_run_twice():
    """run() / prepare() は入力の df を書き換えず、同じ df で何度でも実行できる"""
    df = create_random_walk_df(seed=1)
    original = df.copy()
    backtester = build_strategy()

    first = backtester.run(df)
    pd.testing.assert_frame_equal(df, original)
    backtester.prepare(df)
    pd.testing.assert_frame_equal(df, original)

    second = backtester.run(df)
    pd.testing.assert_series_equal(first.value(), second.value())
    # チャートの出力用に、インジケーターの列が入ったコピーを残す
    assert backtester.df is not None
    assert {"short_ma", "long_ma", "atr"} <= set(backtester.df.columns)
    assert "atr_ratio" not in backtester.df.columns
