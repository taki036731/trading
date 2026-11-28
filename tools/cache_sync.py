"""
株価のキャッシュと JPX の一覧を、private のデータ用リポジトリのリリース資産と共有します（#66）。

使い方:
    python -m tools.cache_sync push  # 手元のキャッシュを zip にまとめてアップロードする（手元の PC で実行）
    python -m tools.cache_sync pull  # zip をダウンロードして展開する（Codespaces などで、必要なときに手動で実行）

このリポジトリは public なので、Yahoo のデータや JPX の一覧はここには置かず、
データ用リポジトリ（DATA_REPO。環境変数 TRADING_DATA_REPO で変えられる）の、
固定のタグ（RELEASE_TAG）のリリースに1つの zip（ASSET_NAME）として上書きで置きます。
アップロード・ダウンロードには GitHub CLI（gh）を使います。手元の PC では `gh auth login` が必要です。
Codespaces では、devcontainer.json でデータ用リポジトリを読む権限を宣言しています
（初めて Codespace を作るときに、権限の承認を求められます）。

pull は zip の中のファイルで上書きし、zip にないファイルは消しません。
展開後は、TRADING_DATA_MODE=offline で yfinance に問い合わせずに実行できます。
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime

from lib import setup_logging
from lib.data_loader import CACHE_DIR
from lib.data_sources import YFinanceSource
from tools.build_universe import SOURCE_DIR

logger = logging.getLogger("tools.cache_sync")

DATA_REPO_ENV = "TRADING_DATA_REPO"
DEFAULT_DATA_REPO = "taki036731/trading-data"
RELEASE_TAG = "cache-latest"
ASSET_NAME = "trading-cache.zip"
MANIFEST_NAME = "manifest.json"

# zip に入れるフォルダ（リポジトリのルートからの相対パス）。展開もこの中にだけ行う
SYNC_DIRS = [os.path.join(CACHE_DIR, YFinanceSource().name), SOURCE_DIR]


def data_repo() -> str:
    """データ用リポジトリ（owner/name）を返します。"""
    return os.environ.get(DATA_REPO_ENV) or DEFAULT_DATA_REPO


def _iter_files(dirs: Iterable[str], root: str) -> list[str]:
    """dirs の中のファイルを、root からの相対パス（区切りは '/'）で返します。"""
    files = []
    for d in dirs:
        base = os.path.join(root, d)
        for dirpath, _, filenames in os.walk(base):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                files.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(files)


def build_archive(zip_path: str, root: str = ".", dirs: Iterable[str] = SYNC_DIRS) -> dict:
    """
    共有するファイルを zip にまとめます。作成日時とファイルの一覧を manifest.json に入れます。

    Args:
        zip_path (str): 作る zip のパス。
        root (str, optional): リポジトリのルート。
        dirs (Iterable[str], optional): zip に入れるフォルダ（root からの相対パス）。

    Returns:
        dict: manifest の内容。

    Raises:
        FileNotFoundError: 入れるファイルが1つもない場合。
    """
    files = _iter_files(dirs, root)
    if not files:
        raise FileNotFoundError(f"共有するファイルがありません: {list(dirs)}")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            zf.write(os.path.join(root, name), name)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def extract_archive(zip_path: str, root: str = ".", dirs: Iterable[str] = SYNC_DIRS) -> dict:
    """
    build_archive で作った zip を展開します。dirs の外へのファイル（'..' を含むものなど）は展開しません。

    Args:
        zip_path (str): zip のパス。
        root (str, optional): リポジトリのルート。
        dirs (Iterable[str], optional): 展開してよいフォルダ（root からの相対パス）。

    Returns:
        dict: manifest の内容。

    Raises:
        ValueError: dirs の外へのファイルが含まれている場合（何も展開しません）。
    """
    root_abs = os.path.abspath(root)
    allowed = [os.path.join(root_abs, os.path.normpath(d)) for d in dirs]
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME))
        members = [m for m in zf.infolist() if m.filename != MANIFEST_NAME and not m.is_dir()]
        for member in members:
            target = os.path.abspath(os.path.join(root_abs, member.filename))
            if not any(os.path.commonpath([target, a]) == a for a in allowed):
                raise ValueError(f"展開できない場所へのファイルが含まれています: {member.filename}")
        for member in members:
            zf.extract(member, root_abs)
    return manifest


def _gh(*args: str) -> subprocess.CompletedProcess:
    """gh コマンドを実行します（失敗したら CalledProcessError）。"""
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True)


def push(repo: str) -> None:
    """キャッシュを zip にまとめ、データ用リポジトリのリリースに上書きでアップロードします。"""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, ASSET_NAME)
        manifest = build_archive(zip_path)
        try:
            _gh("release", "view", RELEASE_TAG, "--repo", repo)
        except subprocess.CalledProcessError:
            logger.info(f"{repo} にリリース {RELEASE_TAG} がないため作ります。")
            _gh(
                "release", "create", RELEASE_TAG,
                "--repo", repo,
                "--title", "株価キャッシュ",
                "--notes", "tools/cache_sync.py が上書きする、株価のキャッシュと JPX の一覧です。",
            )  # fmt: skip
        _gh("release", "upload", RELEASE_TAG, zip_path, "--repo", repo, "--clobber")
        size_mb = os.path.getsize(zip_path) / 1e6
    logger.info(
        f"{len(manifest['files'])} ファイル（{size_mb:.1f} MB）を {repo} の {RELEASE_TAG} に"
        "アップロードしました。"
    )


def pull(repo: str) -> None:
    """データ用リポジトリのリリースから zip をダウンロードし、展開します。"""
    with tempfile.TemporaryDirectory() as tmp:
        _gh(
            "release", "download", RELEASE_TAG,
            "--repo", repo, "--pattern", ASSET_NAME, "--dir", tmp,
        )  # fmt: skip
        manifest = extract_archive(os.path.join(tmp, ASSET_NAME))
    logger.info(
        f"{len(manifest['files'])} ファイルを展開しました"
        f"（{manifest['created_at']} にアップロードしたもの）。"
    )


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="株価のキャッシュをデータ用リポジトリと共有します。")
    parser.add_argument("command", choices=["push", "pull"])
    parser.add_argument("--repo", default=None, help=f"データ用リポジトリ（既定値: {DEFAULT_DATA_REPO}）")
    args = parser.parse_args()

    repo = args.repo or data_repo()
    try:
        if args.command == "push":
            push(repo)
        else:
            pull(repo)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"gh コマンドが失敗しました: {' '.join(e.cmd)}\n{e.stderr}") from None
