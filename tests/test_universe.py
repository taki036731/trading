import pandas as pd
import pytest

from lib.universe import (
    DELISTED_COLUMN,
    UNIVERSE_COLUMNS,
    attach_attributes,
    has_survivorship_bias,
    load_universe_file,
    universe_path,
)
from tools.build_universe import (
    SIZE_QUOTAS,
    SOURCE_PATH,
    build_universe,
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


def test_committed_universe_matches_builder():
    """
    commit 済みのユニバースが、tools/build_universe.py の今の設定で作り直したものと一致することをテスト
    （どう選んだかを再現できるようにするため）
    """
    universe = load_universe_file(universe_path("jpx_stratified_100"))
    rebuilt = build_universe(load_jpx_list(SOURCE_PATH), SIZE_QUOTAS, seed=13)

    assert len(universe) == sum(SIZE_QUOTAS.values())
    pd.testing.assert_frame_equal(universe, rebuilt.astype(str))
