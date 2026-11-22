import numpy as np
import pandas as pd


def create_base_df(
    days: int = 30, start_date: str = "2024-01-01", base_price: float = 1000.0
) -> pd.DataFrame:
    """
    テスト用のベースとなるDataFrameを作成する。
    Open=Close, High=Close+10, Low=Close-10 (ATR=20相当)
    """
    dates = pd.date_range(start_date, periods=days)
    # 緩やかな右肩上がり
    close = np.linspace(base_price, base_price + days * 2, days)

    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
        },
        index=dates,
    )

    # 期待されるカラムを初期化 (0だとシグナルに影響する場合があるのでNoneや適当な値)
    df["short_ma"] = base_price - 10
    df["long_ma"] = base_price
    df["atr"] = 20.0

    return df


def create_random_walk_df(
    days: int = 300, seed: int = 0, start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """
    インジケーター計算から通しで動かすテスト用に、ランダムウォークの OHLC を作る。
    営業日（土日除く）のインデックスで、指標の列は含まない。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=days)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    open_ = close * (1 + rng.normal(0, 0.005, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, days))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=dates
    )


def daily_close_value(portfolio) -> pd.Series:
    """
    Portfolio の資産推移を、日付ごとの最後の行（その日の終値時点）にそろえて返す。

    足を分けた 12:00 の行の有無や、1行目（始値の時点）の評価額の違いに左右されずに、
    分け方の異なる Portfolio の資産推移を比べるために使う。
    """
    value = portfolio.value()
    index = pd.DatetimeIndex(value.index)
    return value.groupby(index.normalize()).last()


def assert_same_daily_result(actual, expected) -> None:
    """
    2つの Portfolio の、日付ごとの終値時点の資産推移とトレードが一致することを確かめる。

    資金共有モード（MultiBacktester.run_portfolio）は、買いのある日を全て2行に分けるため、
    1銘柄だけでも Backtester.run と行数や始値の時点の評価額が変わる。売買の結果が同じことは、これで比べる。
    """
    np.testing.assert_allclose(
        daily_close_value(actual).to_numpy().ravel(), daily_close_value(expected).to_numpy().ravel()
    )
    columns = ["Size", "Avg Entry Price", "Avg Exit Price", "PnL", "Status"]
    actual_trades = actual.trades.records_readable
    expected_trades = expected.trades.records_readable
    pd.testing.assert_frame_equal(
        actual_trades[columns].reset_index(drop=True),
        expected_trades[columns].reset_index(drop=True),
    )
    for col in ["Entry Timestamp", "Exit Timestamp"]:
        assert (
            pd.to_datetime(actual_trades[col]).dt.normalize().tolist()
            == pd.to_datetime(expected_trades[col]).dt.normalize().tolist()
        )
