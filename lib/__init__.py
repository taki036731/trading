import json
import logging
import logging.config
import os


def setup_logging():
    """ログ設定をJSONから読み込んで適用します。"""
    # libディレクトリの親ディレクトリ（プロジェクトルート）を取得
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "logging_config.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    except FileNotFoundError:
        # 万が一設定ファイルが見つからない場合のフォールバック設定
        logging.basicConfig(level=logging.INFO)
        logging.error(f"ログ設定ファイルが見つかりません: {config_path}")
