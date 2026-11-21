from collections.abc import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], owner: object) -> None:
    """
    データフレームに必須カラムが揃っているかを検証します。
    indicator / signal / exit / sizer / 取得元の `generate()` や `fetch()` で共通に使います。

    Args:
        df (pd.DataFrame): 検証するデータフレーム。
        required (Iterable[str]): 必須カラム名。
        owner (object): 検証を行うオブジェクト。エラーメッセージにクラス名を出すために使います。

    Raises:
        ValueError: 必須カラムが1つでも不足している場合。不足しているカラムをまとめて表示します。
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        names = ", ".join(f"'{col}'" for col in missing)
        raise ValueError(f"エラー: {type(owner).__name__} の必須カラム {names} が存在しません。")
