import logging
import os

import gspread
import pandas as pd
from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe

_logger = logging.getLogger(__name__)

# 内部状態
_spreadsheet = None


def _initialize():
    """
    Googleスプレッドシートへの接続を初期化します。

    .envファイルから環境変数を読み込み、サービスアカウントの認証情報を使用して
    指定されたスプレッドシートを開きます。既に初期化済みの場合はスキップします。

    Raises:
        ValueError: 環境変数 SPREADSHEET_ID が設定されていない場合に発生します。
    """
    global _spreadsheet
    if _spreadsheet is None:
        # .envファイルを読み込む
        load_dotenv()

        # 環境変数を取得
        spreadsheet_id = os.getenv("SPREADSHEET_ID")
        credentials_file = os.getenv(
            "CREDENTIALS_FILE", "credentials.json"
        )  # デフォルト値も指定可能

        if not spreadsheet_id:
            raise ValueError("環境変数 SPREADSHEET_ID が設定されていません。")

        _logger.info("Googleスプレッドシートへの接続を初期化しています...")
        gc = gspread.service_account(filename=credentials_file)
        _spreadsheet = gc.open_by_key(spreadsheet_id)
        _logger.info("接続が完了しました。")


def write_df_to_sheet(sheet_name: str, df: pd.DataFrame):
    """
    指定されたシート名に Pandas DataFrame を書き込みます。

    シートが存在しない場合は新規作成し、既存のデータはすべて消去してから
    DataFrame の内容を書き込みます。

    Args:
        sheet_name (str): 書き込み対象のシート名。
        df (pd.DataFrame): 書き込むデータを含む DataFrame。

    Raises:
        ValueError: スプレッドシートの初期化に失敗した場合に発生します。
    """
    _initialize()

    if _spreadsheet is None:
        _initialize()
    if _spreadsheet is None:
        raise ValueError("Spreadsheet has not been initialized.")

    try:
        worksheet = _spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = _spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=10)

    if not worksheet:
        _logger.error("シートが作成できませんでした。")

    worksheet.clear()
    set_with_dataframe(worksheet, df)
    _logger.info(f"'{sheet_name}' シートの更新が完了しました。")
