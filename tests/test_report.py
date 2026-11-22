"""
バックテスト結果の出力（lib/report.py）のテスト（#44）。

注文の日時が分かるように、Backtester を通さずに vbt.Portfolio を直接作って確かめる。
"""

import pandas as pd
import pytest
import vectorbt as vbt

import lib.data_saver as ds
from lib import report

DAYS = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-09"])
# エントリー当日に足を分けた日（2日目）の 12:00 の行
SPLIT_ROW = DAYS[1] + pd.Timedelta(hours=12)


@pytest.fixture
def daily_df() -> pd.DataFrame:
    """日次の株価データ（インジケーターの列を1つ含む）"""
    return pd.DataFrame(
        {"Close": [100.0, 110.0, 120.0], "short_ma": [1.0, 2.0, 3.0]}, index=DAYS
    )


@pytest.fixture
def split_portfolio() -> vbt.Portfolio:
    """2日目の始値で買い、同じ日の 12:00 の行で売る（エントリー当日の SL/TP 決済）Portfolio"""
    index = pd.DatetimeIndex([DAYS[0], DAYS[1], SPLIT_ROW, DAYS[2]])
    price = pd.Series([100.0, 110.0, 105.0, 120.0], index=index)
    entries = pd.Series([False, True, False, False], index=index)
    exits = pd.Series([False, False, True, False], index=index)
    return vbt.Portfolio.from_signals(
        price, entries, exits, size=100, init_cash=100_000, freq="D"
    )


@pytest.fixture
def multi_portfolio() -> vbt.Portfolio:
    """2銘柄（A は1日目に買い、B は2日目に買う）の資金共有の Portfolio"""
    price = pd.DataFrame({"A": [100.0, 110.0, 120.0], "B": [50.0, 55.0, 60.0]}, index=DAYS)
    entries = pd.DataFrame({"A": [True, False, False], "B": [False, True, False]}, index=DAYS)
    exits = pd.DataFrame(False, index=DAYS, columns=["A", "B"])
    return vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        size=100,
        init_cash=100_000,
        cash_sharing=True,
        group_by=True,
        freq="D",
    )


def test_chart_table_aggregates_orders_by_date(daily_df, split_portfolio):
    """同じ日の注文（12:00 の行の決済を含む）が、日付ごとに1行にまとまることをテスト"""
    entries = pd.Series([False, True, False], index=DAYS)
    exits = pd.Series(False, index=DAYS)
    table = report.chart_table(split_portfolio, daily_df, entries, exits)

    # 行は日次の df にそろい、12:00 の行は増えない
    assert list(table.index) == list(DAYS)
    assert {"Close", "short_ma", "entry_signal", "exit_signal"} <= set(table.columns)
    assert table.at[DAYS[1], "exec_side"] == "Buy,Sell"
    assert table.at[DAYS[1], "exec_size"] == pytest.approx(200)
    assert table.at[DAYS[1], "exec_value"] == pytest.approx(100 * 110 + 100 * 105)
    assert table.at[DAYS[1], "exec_price"] == pytest.approx((110 + 105) / 2)
    assert table.at[DAYS[1], "entry_signal"]
    # 注文のない日は NaN
    assert pd.isna(table.loc[DAYS[0], "exec_size"])


def test_chart_table_without_signals(daily_df, split_portfolio):
    """シグナルを省略した場合は、シグナルの列を作らないことをテスト"""
    table = report.chart_table(split_portfolio, daily_df)
    assert "entry_signal" not in table.columns
    assert "exit_signal" not in table.columns
    assert table.loc[DAYS[1], "exec_side"] == "Buy,Sell"


def test_chart_table_filters_column(daily_df, multi_portfolio):
    """複数銘柄の Portfolio で、column に指定した銘柄の注文だけを使うことをテスト"""
    table = report.chart_table(multi_portfolio, daily_df, column="B")
    assert pd.isna(table.loc[DAYS[0], "exec_size"])
    assert table.loc[DAYS[1], "exec_side"] == "Buy"
    assert table.loc[DAYS[1], "exec_price"] == pytest.approx(55.0)


def test_save_csv(tmp_path, split_portfolio):
    """注文履歴・取引履歴を CSV に保存できることをテスト"""
    orders_path = tmp_path / "orders.csv"
    trades_path = tmp_path / "trades.csv"
    report.save_orders_csv(split_portfolio, str(orders_path))
    report.save_trades_csv(split_portfolio, str(trades_path))

    assert len(pd.read_csv(orders_path)) == 2
    trades = pd.read_csv(trades_path)
    assert len(trades) == 1
    assert trades["Size"].iloc[0] == pytest.approx(100)


def test_save_to_sheet(monkeypatch, daily_df, split_portfolio):
    """Sheets の関数が、シート名と表を data_saver に渡すことをテスト"""
    written: dict[str, tuple[pd.DataFrame, bool]] = {}

    def fake_write(sheet_name: str, df: pd.DataFrame, include_index: bool = False):
        written[sheet_name] = (df, include_index)

    monkeypatch.setattr(ds, "write_df_to_sheet", fake_write)
    report.save_chart_to_sheet(split_portfolio, daily_df)
    report.save_orders_to_sheet(split_portfolio)
    report.save_trades_to_sheet(split_portfolio, sheet_name="MyTrades")

    assert set(written) == {"Chart", "Orders", "MyTrades"}
    chart, include_index = written["Chart"]
    assert include_index
    # 空のセルは "" で埋める
    assert chart.loc[DAYS[0], "exec_size"] == ""
    assert len(written["Orders"][0]) == 2
    assert len(written["MyTrades"][0]) == 1


def test_save_table_to_sheet(monkeypatch):
    """任意の表を保存でき、NaN と ±inf は空欄になり、元の表は変わらないことをテスト（#64）"""
    written: dict[str, tuple[pd.DataFrame, bool]] = {}

    def fake_write(sheet_name: str, df: pd.DataFrame, include_index: bool = False):
        written[sheet_name] = (df, include_index)

    monkeypatch.setattr(ds, "write_df_to_sheet", fake_write)
    table = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "profit_factor": [1.5, float("inf"), float("nan")],
            "num_trades": [3, 0, -1],
            "worst": [-0.1, float("-inf"), 0.2],
        }
    )
    report.save_table_to_sheet(table, "Multi_Summary")

    exported, include_index = written["Multi_Summary"]
    assert not include_index
    assert exported["profit_factor"].tolist() == [1.5, "", ""]
    assert exported["worst"].tolist() == [-0.1, "", 0.2]
    assert exported["num_trades"].tolist() == [3, 0, -1]
    assert exported["ticker"].tolist() == ["A", "B", "C"]
    # 呼び出し元の表はそのまま
    assert table["profit_factor"].iloc[1] == float("inf")


@pytest.mark.parametrize(("value", "expected"), [("sheet-id", True), ("", False), (None, False)])
def test_sheet_is_configured(monkeypatch, value, expected):
    """SPREADSHEET_ID の有無で、出力先が設定されているかを返すことをテスト（#64）"""
    # .env の内容に左右されないように、読み込みを止める
    monkeypatch.setattr(ds, "load_dotenv", lambda: None)
    if value is None:
        monkeypatch.delenv("SPREADSHEET_ID", raising=False)
    else:
        monkeypatch.setenv("SPREADSHEET_ID", value)
    assert report.sheet_is_configured() is expected


def test_print_summary(capsys, split_portfolio):
    """結果サマリと最新の注文履歴を表示できることをテスト"""
    report.print_summary(split_portfolio, n_orders=1)
    out = capsys.readouterr().out
    assert "【バックテスト結果サマリ】" in out
    assert "【最新の注文履歴】" in out

