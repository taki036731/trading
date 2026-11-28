import json
import zipfile

import pytest

from tools.cache_sync import MANIFEST_NAME, build_archive, extract_archive

DIRS = ["data/yfinance", "universes/source"]


def _make_tree(root):
    (root / "data" / "yfinance").mkdir(parents=True)
    (root / "data" / "yfinance" / "7203.T.parquet").write_bytes(b"parquet")
    (root / "data" / "yfinance" / "7203.T.meta.json").write_text('{"fetched_at": "x"}')
    (root / "universes" / "source").mkdir(parents=True)
    (root / "universes" / "source" / "data_j_20261031.csv").write_text("日付\n")
    # 共有しないファイル
    (root / "data" / "app.log").write_text("log")


def test_archive_round_trip(tmp_path):
    """共有するフォルダのファイルだけを zip にまとめ、別の場所に同じ構成で展開できることをテスト"""
    src = tmp_path / "src"
    _make_tree(src)
    zip_path = tmp_path / "cache.zip"

    manifest = build_archive(str(zip_path), root=str(src), dirs=DIRS)

    assert manifest["files"] == [
        "data/yfinance/7203.T.meta.json",
        "data/yfinance/7203.T.parquet",
        "universes/source/data_j_20261031.csv",
    ]
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "data" / "yfinance").mkdir(parents=True)
    (dst / "data" / "yfinance" / "6758.T.parquet").write_bytes(b"keep")
    extract_archive(str(zip_path), root=str(dst), dirs=DIRS)

    assert (dst / "data" / "yfinance" / "7203.T.parquet").read_bytes() == b"parquet"
    assert (dst / "universes" / "source" / "data_j_20261031.csv").exists()
    assert not (dst / "data" / "app.log").exists()
    assert not (dst / MANIFEST_NAME).exists()
    # zip にないファイルは消さない
    assert (dst / "data" / "yfinance" / "6758.T.parquet").read_bytes() == b"keep"


def test_build_archive_raises_without_files(tmp_path):
    """共有するファイルが1つもなければエラーにする（空の zip で上書きしないように）"""
    with pytest.raises(FileNotFoundError):
        build_archive(str(tmp_path / "cache.zip"), root=str(tmp_path), dirs=DIRS)


@pytest.mark.parametrize("name", ["../evil.txt", "lib/evil.py", "data/yfinance/../../evil.txt"])
def test_extract_archive_rejects_outside_paths(tmp_path, name):
    """共有するフォルダの外へのファイルを含む zip は、何も展開せずにエラーにする"""
    zip_path = tmp_path / "cache.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/yfinance/ok.parquet", b"ok")
        zf.writestr(name, b"evil")
        zf.writestr(MANIFEST_NAME, json.dumps({"created_at": "x", "files": []}))
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(ValueError, match="展開できない"):
        extract_archive(str(zip_path), root=str(root), dirs=DIRS)
    assert not (root / "data" / "yfinance" / "ok.parquet").exists()


def test_create_release_explains_empty_repository(monkeypatch):
    """空のリポジトリでリリースを作れないときは、README を足すよう案内して止める"""
    import subprocess

    from tools import cache_sync

    def fake_gh(*args):
        raise subprocess.CalledProcessError(
            1, ["gh", *args], stderr="HTTP 422: Validation Failed\nRepository is empty."
        )

    monkeypatch.setattr(cache_sync, "_gh", fake_gh)

    with pytest.raises(SystemExit, match="README"):
        cache_sync._create_release("owner/repo")
