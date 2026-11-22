"""
JPX の「東証上場銘柄一覧」から、業種・規模で層化して銘柄を選び、ユニバースの CSV を作ります（#13）。

使い方:
    python -m tools.build_universe

入力は JPX が公開している一覧（data_j.xls）を UTF-8 の CSV に変換したもの（SOURCE_PATH）です。
対象は TOPIX の構成銘柄（規模区分のある銘柄）で、規模区分ごとに SIZE_QUOTAS の数を選びます。
規模区分の中では、銘柄を固定の乱数シードで並べ替えてから33業種の順に並べ、等間隔に選ぶので、
業種の構成比がおおよそ元の一覧と同じになります。同じ入力とシードなら、同じ銘柄が選ばれます。

一覧は作成した日に上場している銘柄だけなので、このユニバースには生存者バイアスがあります。
"""

import logging
import os

import numpy as np
import pandas as pd

from lib import setup_logging
from lib.universe import UNIVERSE_COLUMNS, universe_path

logger = logging.getLogger("tools.build_universe")

SOURCE_PATH = os.path.join("universes", "source", "data_j_20261031.csv")
UNIVERSE_NAME = "jpx_stratified_100"
SEED = 13

# 規模区分ごとに選ぶ銘柄数（合計100）。大型株に偏らないよう、中小型株を多めにしている
SIZE_QUOTAS = {
    "TOPIX Core30": 10,
    "TOPIX Large70": 15,
    "TOPIX Mid400": 30,
    "TOPIX Small 1": 25,
    "TOPIX Small 2": 20,
}

# JPX の一覧の列名 → ユニバースの列名
SOURCE_COLUMNS = {
    "コード": "code",
    "銘柄名": "name",
    "市場・商品区分": "market",
    "33業種区分": "sector",
    "規模区分": "size",
    "日付": "listed_as_of",
}


def load_jpx_list(path: str) -> pd.DataFrame:
    """
    JPX の東証上場銘柄一覧（UTF-8 の CSV）を読み込み、列名をそろえます。

    Args:
        path (str): 一覧の CSV のパス。

    Returns:
        pd.DataFrame: 列は SOURCE_COLUMNS の値（全て文字列）。
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)


def stratified_sample(stocks: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    銘柄を33業種の構成比に沿って n 件選びます。

    固定のシードで並べ替えてから業種の順に（業種の中はその並びのまま）並べ、等間隔に選びます。

    Args:
        stocks (pd.DataFrame): sector 列を持つ銘柄の一覧。
        n (int): 選ぶ件数。一覧の件数より多い場合は全件を返します。
        seed (int): 乱数シード。

    Returns:
        pd.DataFrame: 選んだ銘柄。
    """
    if n >= len(stocks):
        return stocks
    shuffled = stocks.sample(frac=1.0, random_state=seed)
    ordered = shuffled.sort_values("sector", kind="stable")
    positions = ((np.arange(n) + 0.5) * len(ordered) / n).astype(int)
    return ordered.iloc[positions]


def build_universe(source: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    """
    TOPIX の構成銘柄から、規模区分ごとに quotas の数を業種で層化して選びます。

    Args:
        source (pd.DataFrame): load_jpx_list の戻り値。
        quotas (dict[str, int]): 規模区分ごとの件数。
        seed (int): 乱数シード。

    Returns:
        pd.DataFrame: ユニバース。列は UNIVERSE_COLUMNS と market / listed_as_of。
            並びは quotas の規模区分の順、その中は業種の順。
    """
    stocks = source[source["size"].isin(list(quotas))]
    parts = [
        stratified_sample(stocks[stocks["size"] == size], n, seed) for size, n in quotas.items()
    ]
    universe = pd.concat(parts)
    universe = universe.assign(ticker=universe["code"] + ".T")
    return universe[[*UNIVERSE_COLUMNS, "market", "listed_as_of"]].reset_index(drop=True)


if __name__ == "__main__":
    setup_logging()
    source = load_jpx_list(SOURCE_PATH)
    universe = build_universe(source, SIZE_QUOTAS, SEED)

    path = universe_path(UNIVERSE_NAME)
    universe.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"{len(universe)} 銘柄のユニバースを {path} に保存しました。")
    print(universe.groupby("size", sort=False).size())
    print(universe["sector"].value_counts())
