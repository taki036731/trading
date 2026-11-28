"""
JPX の「東証上場銘柄一覧」から、業種・規模で層化して銘柄を選び、ユニバースの CSV を作ります（#13）。

使い方:
    python -m tools.build_universe             # SOURCE_PATH の一覧からユニバースを作る
    python -m tools.build_universe --download  # JPX から最新の一覧をダウンロードして CSV にする（#66）

入力は JPX が公開している一覧（data_j.xlsx）を UTF-8 の CSV に変換したもの（SOURCE_PATH）です。
一覧は毎月第3営業日に前月末の版に差し替わります。`--download` は、それを
`universes/source/data_j_<一覧の日付>.csv` に保存するだけで、ユニバースは作り直しません
（新しい一覧で作り直すときは、SOURCE_PATH を書き換えてから実行し、ユニバースを commit します）。
JPX の利用規約は、一覧の許諾のない二次利用・再配信を禁じているため、一覧（universes/source/）は
このリポジトリ（public）には commit せず、tools/cache_sync.py で private のデータ用リポジトリと共有します。
規約は高頻度の自動取得も控えるよう求めているので、ダウンロードは手動で月1回程度にとどめます。
対象は TOPIX の構成銘柄（規模区分のある銘柄）で、規模区分ごとに SIZE_QUOTAS の数を選びます。
規模区分の中では、銘柄を固定の乱数シードで並べ替えてから33業種の順に並べ、等間隔に選ぶので、
業種の構成比がおおよそ元の一覧と同じになります。同じ入力とシードなら、同じ銘柄が選ばれます。

一覧は作成した日に上場している銘柄だけなので、このユニバースには生存者バイアスがあります。
"""

import argparse
import io
import logging
import os

import numpy as np
import pandas as pd
import requests

from lib import setup_logging
from lib.universe import UNIVERSE_COLUMNS, universe_path

logger = logging.getLogger("tools.build_universe")

SOURCE_DIR = os.path.join("universes", "source")
SOURCE_PATH = os.path.join(SOURCE_DIR, "data_j_20261031.csv")
# JPX の「東証上場銘柄一覧」のページ（https://www.jpx.co.jp/markets/statistics-equities/misc/01.html）の Excel。
# リンクが変わってダウンロードできなくなったら、ページで確認して書き換える
JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
)
DOWNLOAD_TIMEOUT = 60
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


def _to_text(value: object) -> str:
    """Excel のセルの値を、CSV と同じ文字列にします（整数の 50.0 は '50'、空のセルは ''）。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def convert_jpx_excel(content: bytes) -> pd.DataFrame:
    """
    JPX の一覧の Excel（data_j.xlsx）を、load_jpx_list で読める表（全て文字列）にします。

    Args:
        content (bytes): Excel ファイルの中身。

    Returns:
        pd.DataFrame: 列は Excel のまま（日付, コード, 銘柄名, ...）。

    Raises:
        ValueError: 必要な列（SOURCE_COLUMNS のキー）がない場合。
    """
    df = pd.read_excel(io.BytesIO(content), dtype=object, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"JPX の一覧に必要な列がありません: {missing}（列: {list(df.columns)}）")
    return df.map(_to_text)


def download_jpx_list(url: str = JPX_LIST_URL, out_dir: str = SOURCE_DIR) -> str:
    """
    JPX の一覧をダウンロードし、`<out_dir>/data_j_<一覧の日付>.csv`（UTF-8）に保存します。

    Args:
        url (str, optional): 一覧の Excel の URL。
        out_dir (str, optional): 保存先のフォルダ。

    Returns:
        str: 保存した CSV のパス。
    """
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    df = convert_jpx_excel(response.content)

    dates = df["日付"].unique()
    if len(dates) != 1:
        raise ValueError(f"JPX の一覧の日付が1つではありません: {list(dates)}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"data_j_{dates[0]}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="JPX から最新の一覧をダウンロードして universes/source/ に保存する（ユニバースは作り直さない）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    if args.download:
        saved = download_jpx_list()
        logger.info(
            f"JPX の一覧を {saved} に保存しました。"
            "この一覧でユニバースを作り直すときは、SOURCE_PATH を書き換えてから実行してください。"
        )
        raise SystemExit(0)

    if not os.path.exists(SOURCE_PATH):
        raise SystemExit(
            f"一覧 {SOURCE_PATH} がありません。"
            "`python -m tools.cache_sync pull` で取得するか、`--download` でダウンロードしてください。"
        )
    source = load_jpx_list(SOURCE_PATH)
    universe = build_universe(source, SIZE_QUOTAS, SEED)

    path = universe_path(UNIVERSE_NAME)
    universe.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"{len(universe)} 銘柄のユニバースを {path} に保存しました。")
    print(universe.groupby("size", sort=False).size())
    print(universe["sector"].value_counts())
