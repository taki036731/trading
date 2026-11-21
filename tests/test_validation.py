import pandas as pd
import pytest

from lib.strategies.exits import AtrStopLoss, AtrTakeProfit
from lib.strategies.indicators import ATRIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import GoldenCrossSignal
from lib.validation import validate_required_columns


def test_validate_passes_when_all_columns_exist():
    """必須カラムが揃っていれば何もしない"""
    df = pd.DataFrame({"a": [1], "b": [2]})

    validate_required_columns(df, ["a", "b"], object())


def test_validate_lists_all_missing_columns_with_class_name():
    """不足しているカラムをまとめて、クラス名つきで表示する"""
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError) as exc:
        validate_required_columns(df, ["a", "b", "c"], GoldenCrossSignal())

    message = str(exc.value)
    assert "GoldenCrossSignal" in message
    assert "'b'" in message
    assert "'c'" in message
    assert "'a'" not in message


@pytest.mark.parametrize(
    ("component", "call"),
    [
        (ATRIndicator(), lambda c, df: c.generate(df)),
        (GoldenCrossSignal(), lambda c, df: c.generate(df)),
        (AtrStopLoss(2), lambda c, df: c.generate(df)),
        (AtrTakeProfit(4), lambda c, df: c.generate(df)),
        (RiskPercentageSizer(), lambda c, df: c.generate(df, sl_pct=pd.Series(dtype=float))),
    ],
)
def test_components_raise_on_missing_columns(component, call):
    """各部品の generate() は、必須カラムがなければクラス名つきのエラーにする"""
    df = pd.DataFrame({"Open": [1.0]})

    with pytest.raises(ValueError, match=type(component).__name__):
        call(component, df)
