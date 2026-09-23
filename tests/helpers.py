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
