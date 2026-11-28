"""
検証する銘柄ユニバース（銘柄の一覧と、業種・規模などの属性）をファイルで管理します。

ユニバースは `universes/<名前>.csv` に置き、列は UNIVERSE_COLUMNS です。
株価データの取得は lib.multi_backtester.load_universe に ticker 列を渡して行います。
"""

import logging
import os

import pandas as pd

from lib.validation import validate_required_columns

logger = logging.getLogger(__name__)

UNIVERSE_DIR = "universes"

# ユニバースのファイルの必須列
# - ticker: yfinance のティッカー（例: 7203.T）
# - name: 銘柄名
# - sector: 東証33業種
# - size: 規模区分（TOPIX Core30 / Large70 / Mid400 / Small 1 / Small 2）
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "size"]

# 銘柄ごとにデータを使い始める日の列（任意。空なら全期間を使う）。#70
# 上場前のデータが壊れている銘柄（同じコードの別の銘柄のデータが入っているなど）で、上場日などを入れる
DATA_FROM_COLUMN = "data_from"

# 上場廃止日の列。上場廃止した銘柄を含むユニバースで使う（#13 の第2段階）
DELISTED_COLUMN = "delisted_on"

SURVIVORSHIP_NOTE = (
    "このユニバースは、作成した時点で上場している銘柄だけを含みます（生存者バイアス）。"
    "上場廃止・低迷した銘柄が入らないため、結果は実際より良く見えやすい点に注意してください。"
)


class _UniverseFile:
    """validate_required_columns のエラーメッセージに出す名前のためのクラスです。"""


def universe_path(name: str) -> str:
    """
    ユニバースの名前から、ファイルのパスを返します。

    Args:
        name (str): ユニバースの名前（例: 'jpx_stratified_100'）。

    Returns:
        str: `universes/<名前>.csv` のパス。
    """
    return os.path.join(UNIVERSE_DIR, f"{name}.csv")


def load_universe_file(path: str) -> pd.DataFrame:
    """
    ユニバースのファイル（CSV、UTF-8）を読み込みます。

    Args:
        path (str): ユニバースの CSV のパス。

    Returns:
        pd.DataFrame: ファイルの全ての列（ticker が先頭。全て文字列）。行の順序はファイルのまま
            （資金共有モードで、同じサイズの買いが重なったときの順序になる）。

    Raises:
        ValueError: 必須列が不足している場合、ticker が空・重複している場合。
    """
    universe = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_required_columns(universe, UNIVERSE_COLUMNS, _UniverseFile())

    universe["ticker"] = universe["ticker"].str.strip()
    if (universe["ticker"] == "").any():
        raise ValueError(f"エラー: ユニバース {path} に ticker が空の行があります。")
    duplicated = universe["ticker"][universe["ticker"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"エラー: ユニバース {path} の ticker が重複しています。：{duplicated}")
    return universe


def data_from_map(universe: pd.DataFrame) -> dict[str, str]:
    """
    ユニバースの data_from 列から、銘柄ごとのデータを使い始める日の辞書を作ります。

    lib.multi_backtester.load_universe の data_from 引数に渡します。列がない、または空の銘柄は含みません。

    Args:
        universe (pd.DataFrame): load_universe_file の戻り値。

    Returns:
        dict[str, str]: ティッカーをキー、'YYYY-MM-DD' の日付を値とする辞書。

    Raises:
        ValueError: 日付として読めない値がある場合。
    """
    if DATA_FROM_COLUMN not in universe.columns:
        return {}
    result: dict[str, str] = {}
    for ticker, value in zip(universe["ticker"], universe[DATA_FROM_COLUMN].fillna("")):
        text = str(value).strip()
        if not text:
            continue
        try:
            result[ticker] = pd.Timestamp(text).strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"エラー: ユニバースの {DATA_FROM_COLUMN} が日付ではありません。：{ticker} {text!r}"
            ) from None
    return result


def has_survivorship_bias(universe: pd.DataFrame) -> bool:
    """
    ユニバースに生存者バイアスがあるか（上場廃止した銘柄を含まないか）を判定します。

    上場廃止日の列（DELISTED_COLUMN）がない、または全て空なら、作成時点で上場している銘柄だけと見なします。

    Args:
        universe (pd.DataFrame): load_universe_file の戻り値。

    Returns:
        bool: 上場廃止した銘柄を1つも含まなければ True。
    """
    if DELISTED_COLUMN not in universe.columns:
        return True
    return bool((universe[DELISTED_COLUMN].fillna("").str.strip() == "").all())


def attach_attributes(
    summary: pd.DataFrame, universe: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    run_independent の結果に、ユニバースの属性（業種・規模など）の列を ticker で結合します。

    MultiBacktester.aggregate(summary, by=[...]) で、業種・規模ごとに集計するために使います。

    Args:
        summary (pd.DataFrame): ticker 列を持つ結果（run_independent の戻り値）。
        universe (pd.DataFrame): load_universe_file の戻り値。
        columns (list[str] | None): 結合する列。None の場合は sector と size。

    Returns:
        pd.DataFrame: summary の行の順序のまま、属性の列を足した新しい DataFrame。
            ユニバースにない ticker の属性は NaN になります。
    """
    columns = columns or ["sector", "size"]
    validate_required_columns(universe, ["ticker", *columns], _UniverseFile())
    return summary.merge(universe[["ticker", *columns]], on="ticker", how="left")
