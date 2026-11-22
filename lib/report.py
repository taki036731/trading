"""
バックテスト結果（vbt.Portfolio）の出力（表示・CSV・Google Sheets・チャート）。

Backtester はシミュレーションに専念させ、出力はここの関数で行います（#44）。
どの関数も vbt.Portfolio を受け取るので、Backtester.run() の結果にも、
MultiBacktester.run_portfolio() の結果（列=銘柄）にも使えます。

Google Sheets への書き込みは lib.data_saver（gspread）に依存します。
表示や CSV だけを使うときに gspread や .env が要らないように、Sheets の関数の中で import します。
"""

import logging

import pandas as pd
import vectorbt as vbt

from lib.metrics import portfolio_stats
from lib.universe import SURVIVORSHIP_NOTE, has_survivorship_bias

logger = logging.getLogger(__name__)


def orders_table(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """
    注文履歴を、読みやすい形式の DataFrame で返します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。

    Returns:
        pd.DataFrame: 注文履歴（orders.records_readable）。
    """
    # vectorbt の cached_property を Pylance が関数と見なすため、このルールだけ無視する
    return portfolio.orders.records_readable  # pyright: ignore[reportFunctionMemberAccess]


def trades_table(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """
    取引履歴（エントリーからエグジットまでの一連の結果）を、読みやすい形式の DataFrame で返します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。

    Returns:
        pd.DataFrame: 取引履歴（trades.records_readable）。
    """
    # vectorbt の cached_property を Pylance が関数と見なすため、このルールだけ無視する
    return portfolio.trades.records_readable  # pyright: ignore[reportFunctionMemberAccess]


def print_summary(portfolio: vbt.Portfolio, n_orders: int = 5) -> None:
    """
    バックテストの結果サマリと最新の注文履歴を表示します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        n_orders (int): 表示する最新の注文の件数。0 の場合は注文履歴を表示しません。
    """
    print("-" * 40)
    print("【バックテスト結果サマリ】")
    # 勝率、プロフィットファクター、ドローダウンなどが一括で計算されます。
    # 年率換算の基準は MultiBacktester の calc_metrics とそろえる
    print(portfolio_stats(portfolio))
    print("-" * 40)
    if n_orders > 0:
        print("【最新の注文履歴】")
        print(orders_table(portfolio).tail(n_orders))
        print("-" * 40)


def print_universe_note(universe: pd.DataFrame, name: str) -> None:
    """
    検証したユニバースの名前・銘柄数と、生存者バイアスの注意書きを表示します。

    結果を読むときに、どの銘柄で検証したかと、結果の限界が分かるように、集計の前に表示します。

    Args:
        universe (pd.DataFrame): lib.universe.load_universe_file の戻り値。
        name (str): ユニバースの名前（ファイル名など）。
    """
    print("-" * 40)
    print(f"【ユニバース】{name}（{len(universe)} 銘柄）")
    if has_survivorship_bias(universe):
        print(f"※ {SURVIVORSHIP_NOTE}")
        logger.warning(f"[{name}] {SURVIVORSHIP_NOTE}")
    print("-" * 40)


def save_orders_csv(
    portfolio: vbt.Portfolio, path: str = "vectorbt_orders.csv"
) -> None:
    """
    注文履歴を CSV 形式で保存します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        path (str): 保存するファイル名。デフォルトは 'vectorbt_orders.csv'。
    """
    orders_table(portfolio).to_csv(path)
    logger.info(f"注文履歴を {path} に保存しました。")


def save_trades_csv(portfolio: vbt.Portfolio, path: str = "vectorbt_trades.csv") -> None:
    """
    取引履歴を CSV 形式で保存します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        path (str): 保存するファイル名。デフォルトは 'vectorbt_trades.csv'。
    """
    trades_table(portfolio).to_csv(path, index=False)
    logger.info(f"取引履歴を {path} に保存しました。")


def chart_table(
    portfolio: vbt.Portfolio,
    df: pd.DataFrame,
    entries: pd.Series | None = None,
    exits: pd.Series | None = None,
    column: str | None = None,
) -> pd.DataFrame:
    """
    価格・指標、シグナル、約定を日付ごとに結合した表を作ります。

    約定は日付ごとに集約し、exec_size（株数の合計）、exec_price（加重平均価格）、
    exec_value（売買代金の合計）、exec_side（Buy / Sell）の列にします。
    エントリー当日に足を分けた日は SL/TP の決済が 12:00 になるため、日付にそろえて集計します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        df (pd.DataFrame): 日次の株価データ（インジケーターの列を含めてよい）。行はこの index にそろえます。
        entries (pd.Series | None): エントリーシグナル。None の場合は entry_signal の列を作りません。
        exits (pd.Series | None): エグジットシグナル。None の場合は exit_signal の列を作りません。
        column (str | None): 複数銘柄の Portfolio のうち、注文を使う列（銘柄）。
            None の場合は全ての列の注文を使います（単一銘柄の Portfolio ではこれでよい）。

    Returns:
        pd.DataFrame: df の列に、entry_signal / exit_signal / exec_* の列を加えた表。
    """
    orders = orders_table(portfolio).copy()
    if column is not None:
        orders = orders[orders["Column"] == column]
    orders["Timestamp"] = pd.to_datetime(orders["Timestamp"]).dt.normalize()
    orders["Value"] = orders["Size"] * orders["Price"]

    # 同じ日に複数注文がある可能性を考慮して集約
    order_summary = orders.groupby("Timestamp").agg(
        {
            "Size": "sum",
            "Value": "sum",
            "Side": lambda x: ",".join(x.astype(str).unique()),
        }
    )
    # 加重平均価格を算出 (Total Value / Total Size)
    order_summary["Price"] = order_summary["Value"] / order_summary["Size"]
    order_summary = order_summary.rename(
        columns={
            "Size": "exec_size",
            "Price": "exec_price",
            "Value": "exec_value",
            "Side": "exec_side",
        }
    )[["exec_size", "exec_price", "exec_value", "exec_side"]]

    parts: list[pd.DataFrame | pd.Series] = [df]
    if entries is not None:
        parts.append(entries.rename("entry_signal"))
    if exits is not None:
        parts.append(exits.rename("exit_signal"))
    # 行は df にそろえる（注文の日付は df の index に含まれる）
    return pd.concat(parts, axis=1).join(order_summary)


def save_chart_to_sheet(
    portfolio: vbt.Portfolio,
    df: pd.DataFrame,
    entries: pd.Series | None = None,
    exits: pd.Series | None = None,
    column: str | None = None,
    sheet_name: str = "Chart",
) -> None:
    """
    chart_table() の表を、スプレッドシートに保存します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        df (pd.DataFrame): 日次の株価データ（インジケーターの列を含めてよい）。
        entries (pd.Series | None): エントリーシグナル。
        exits (pd.Series | None): エグジットシグナル。
        column (str | None): 複数銘柄の Portfolio のうち、注文を使う列（銘柄）。
        sheet_name (str): 保存先のシート名。デフォルトは 'Chart'。
    """
    import lib.data_saver as ds

    export_df = chart_table(portfolio, df, entries, exits, column)
    ds.write_df_to_sheet(sheet_name, export_df.fillna(""), include_index=True)
    logger.info(f"売買履歴を含む時系列データを '{sheet_name}' シートに保存しました。")


def save_orders_to_sheet(portfolio: vbt.Portfolio, sheet_name: str = "Orders") -> None:
    """
    注文履歴をスプレッドシートに保存します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        sheet_name (str): 保存先のシート名。デフォルトは 'Orders'。
    """
    import lib.data_saver as ds

    ds.write_df_to_sheet(sheet_name, orders_table(portfolio), include_index=False)
    logger.info(f"注文履歴を '{sheet_name}' シートに保存しました。")


def save_trades_to_sheet(portfolio: vbt.Portfolio, sheet_name: str = "Trades") -> None:
    """
    取引履歴をスプレッドシートに保存します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
        sheet_name (str): 保存先のシート名。デフォルトは 'Trades'。
    """
    import lib.data_saver as ds

    ds.write_df_to_sheet(sheet_name, trades_table(portfolio), include_index=False)
    logger.info(f"取引履歴を '{sheet_name}' シートに保存しました。")


def show(portfolio: vbt.Portfolio) -> None:
    """
    バックテストの結果をチャートで表示します。

    Args:
        portfolio (vbt.Portfolio): バックテスト結果。
    """
    fig = portfolio.plot()
    if fig is None:
        logger.warning("表示できるチャートがありません。")
        return
    fig.show()
