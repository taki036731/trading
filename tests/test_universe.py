import os

import numpy as np
import pandas as pd
import pytest

from lib.universe import (
    DATA_FROM_COLUMN,
    DELISTED_COLUMN,
    UNIVERSE_COLUMNS,
    attach_attributes,
    data_from_map,
    has_survivorship_bias,
    load_universe_file,
    universe_path,
)
from tools.build_universe import (
    DATA_FROM_OVERRIDES,
    SIZE_QUOTAS,
    SOURCE_PATH,
    SOURCE_COLUMNS,
    build_universe,
    convert_jpx_excel,
    load_jpx_list,
    stratified_sample,
)


def _write_csv(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "universe.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return str(path)


def _row(ticker: str, sector: str = "輸送用機器", size: str = "TOPIX Core30") -> dict:
    return {"ticker": ticker, "name": f"銘柄{ticker}", "sector": sector, "size": size}


def test_load_universe_file_keeps_order(tmp_path):
    path = _write_csv(tmp_path, [_row("7203.T"), _row(" 6758.T "), _row("130A.T")])
    universe = load_universe_file(path)
    # 行の順序は、資金共有モードで買いが重なったときの順序になるので変えない
    assert universe["ticker"].tolist() == ["7203.T", "6758.T", "130A.T"]
    assert set(UNIVERSE_COLUMNS) <= set(universe.columns)


def test_load_universe_file_missing_columns(tmp_path):
    path = _write_csv(tmp_path, [{"ticker": "7203.T", "name": "トヨタ"}])
    with pytest.raises(ValueError, match="'sector', 'size'"):
        load_universe_file(path)


def test_load_universe_file_duplicated_ticker(tmp_path):
    path = _write_csv(tmp_path, [_row("7203.T"), _row("6758.T"), _row("7203.T")])
    with pytest.raises(ValueError, match="重複"):
        load_universe_file(path)


def test_load_universe_file_empty_ticker(tmp_path):
    path = _write_csv(tmp_path, [_row("7203.T"), _row("")])
    with pytest.raises(ValueError, match="空"):
        load_universe_file(path)


def test_has_survivorship_bias(tmp_path):
    listed_only = load_universe_file(_write_csv(tmp_path, [_row("7203.T"), _row("6758.T")]))
    assert has_survivorship_bias(listed_only)

    empty_delisted = listed_only.assign(**{DELISTED_COLUMN: ["", ""]})
    assert has_survivorship_bias(empty_delisted)

    with_delisted = listed_only.assign(**{DELISTED_COLUMN: ["", "2020-03-31"]})
    assert not has_survivorship_bias(with_delisted)


def test_data_from_map(tmp_path):
    """data_from の空でない銘柄だけを、'YYYY-MM-DD' にそろえて返す（#70）"""
    rows = [
        {**_row("7203.T"), DATA_FROM_COLUMN: ""},
        {**_row("8919.T"), DATA_FROM_COLUMN: "2017/12/12"},
    ]
    universe = load_universe_file(_write_csv(tmp_path, rows))
    assert data_from_map(universe) == {"8919.T": "2017-12-12"}
    # 列がないユニバース（large5 など）は空
    assert data_from_map(load_universe_file(_write_csv(tmp_path, [_row("7203.T")]))) == {}


def test_data_from_map_rejects_invalid_date(tmp_path):
    rows = [{**_row("8919.T"), DATA_FROM_COLUMN: "上場日"}]
    with pytest.raises(ValueError, match="8919.T"):
        data_from_map(load_universe_file(_write_csv(tmp_path, rows)))


def test_committed_universe_has_data_from_overrides():
    """commit 済みのユニバースの data_from が、DATA_FROM_OVERRIDES と一致することをテスト（一覧がなくても確かめる）"""
    universe = load_universe_file(universe_path("jpx_stratified_100"))
    assert data_from_map(universe) == {
        t: d for t, d in DATA_FROM_OVERRIDES.items() if t in set(universe["ticker"])
    }


def test_attach_attributes_keeps_summary_rows():
    summary = pd.DataFrame({"ticker": ["B.T", "A.T", "C.T"], "total_return": [0.1, 0.2, 0.3]})
    universe = pd.DataFrame([_row("A.T", "銀行業", "TOPIX Mid400"), _row("B.T")])
    result = attach_attributes(summary, universe)

    assert result["ticker"].tolist() == ["B.T", "A.T", "C.T"]
    assert result["sector"].tolist()[:2] == ["輸送用機器", "銀行業"]
    # ユニバースにない銘柄は NaN（行は落とさない）
    assert pd.isna(result["sector"].iloc[2])


def test_stratified_sample_follows_sector_ratio():
    stocks = pd.DataFrame(
        {"code": [str(i) for i in range(100)], "sector": ["A"] * 60 + ["B"] * 30 + ["C"] * 10}
    )
    sample = stratified_sample(stocks, 10, seed=0)
    assert sample["sector"].value_counts().to_dict() == {"A": 6, "B": 3, "C": 1}
    assert sample["code"].is_unique
    # 同じシードなら同じ銘柄が選ばれる
    assert sample["code"].tolist() == stratified_sample(stocks, 10, seed=0)["code"].tolist()


@pytest.mark.skipif(
    not os.path.exists(SOURCE_PATH),
    reason="JPX の一覧は public のリポジトリに置けないため、ない環境（CI など）ではスキップする（#66）",
)
def test_committed_universe_matches_builder():
    """
    commit 済みのユニバースが、tools/build_universe.py の今の設定で作り直したものと一致することをテスト
    （どう選んだかを再現できるようにするため）
    """
    universe = load_universe_file(universe_path("jpx_stratified_100"))
    rebuilt = build_universe(load_jpx_list(SOURCE_PATH), SIZE_QUOTAS, seed=13)

    assert len(universe) == sum(SIZE_QUOTAS.values())
    pd.testing.assert_frame_equal(universe, rebuilt.astype(str))


def test_convert_jpx_excel_matches_csv_format(tmp_path):
    """
    JPX の一覧の Excel を、これまで手で変換した CSV と同じ文字列にすることをテスト
    （数値のセルは '50'、'-' や空のセルはそのまま・空文字）
    """
    excel = pd.DataFrame(
        {
            "日付": [20261031, 20261031],
            "コード": [1301, "130A"],
            "銘柄名": ["極洋", "ｉＦｒｅｅＥＴＦ"],
            "市場・商品区分": ["プライム（内国株式）", "ETF・ETN"],
            "33業種コード": [50.0, "-"],
            "33業種区分": ["水産・農林業", "-"],
            "規模コード": [6, np.nan],
            "規模区分": ["TOPIX Small 1", "-"],
        }
    )
    path = tmp_path / "data_j.xlsx"
    excel.to_excel(path, index=False, engine="openpyxl")

    df = convert_jpx_excel(path.read_bytes())

    assert df.loc[0, "日付"] == "20261031"
    assert df["コード"].tolist() == ["1301", "130A"]
    assert df["33業種コード"].tolist() == ["50", "-"]
    assert df["規模コード"].tolist() == ["6", ""]
    # 変換した CSV を、ユニバースを作るときと同じ方法で読める
    csv = tmp_path / "data_j.csv"
    df.to_csv(csv, index=False, encoding="utf-8")
    assert list(load_jpx_list(str(csv)).columns) == list(SOURCE_COLUMNS.values())


def test_convert_jpx_excel_rejects_missing_columns(tmp_path):
    """必要な列がない Excel（形式が変わった場合など）はエラーにする"""
    path = tmp_path / "data_j.xlsx"
    pd.DataFrame({"コード": [1301]}).to_excel(path, index=False, engine="openpyxl")

    with pytest.raises(ValueError, match="銘柄名"):
        convert_jpx_excel(path.read_bytes())
