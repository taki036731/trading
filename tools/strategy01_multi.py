import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401

from lib import report, setup_logging
from lib.backtester import UNIT_SHARES, Backtester
from lib.benchmark import BENCHMARK_TICKER, buy_and_hold, equal_weight_buy_and_hold
from lib.multi_backtester import MultiBacktester, load_universe
from lib.strategies.exits import (
    AtrStopLoss,
    AtrTakeProfit,
)
from lib.strategies.indicators import ATRIndicator, MAIndicator
from lib.strategies.position_sizing import RiskPercentageSizer
from lib.strategies.signals import (
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
)
from lib.universe import attach_attributes, data_from_map, load_universe_file, universe_path

# python -m で実行すると __name__ が "__main__" になるため、tools のロガー設定が効く名前を固定で付ける
logger = logging.getLogger("tools.strategy01_multi")

# 検証する銘柄ユニバース（universes/<名前>.csv。#13）
# - jpx_stratified_100: TOPIX の構成銘柄から、規模区分 × 33業種で層化して選んだ100銘柄（tools/build_universe.py）
# - large5: 以前の検証銘柄（大型株5銘柄）
# 資金共有モードで同じ日に買いが重なった場合は、サイズの小さい順に先着。サイズが同じならファイルの行の順
UNIVERSE_NAME = "jpx_stratified_100"

# 検証するパラメータの候補（全組み合わせを実行する）
PARAM_GRID = {
    "short": [5, 10],
    "long": [20, 50],
    "tp_mult": [3, 4],
}

# 評価期間の開始日。データはインジケーターのウォームアップのために、これより前から取得する（#38）
EVAL_START = "2016-06-25"

# 初期資金（独立モードの各口座と、資金共有モードの口座全体に共通）
# Backtester の既定値（100万円）では、1単元に届かずに捨てられるエントリー注文が多く、
# 戦略の差ではなく「注文が通ったか」の差が結果に出ていたため、1,000万円にしている（#49）
INIT_CASH = 10_000_000.0

# 結果を Google スプレッドシートにも出力するか（#64）。CSV は常に出力する
# 出力先は .env の SPREADSHEET_ID / CREDENTIALS_FILE。未設定なら警告してシートへの出力だけを飛ばす
# strategy01 のシート（Chart / Orders / Trades）を上書きしないように、シート名に接頭辞を付ける
SAVE_TO_SHEET = True
SHEET_PREFIX = "Multi_"


def build_strategy(short: int = 5, long: int = 20, tp_mult: float = 4) -> Backtester:
    """
    strategy01 の戦略を、パラメータを差し替えられる形で作ります。

    Args:
        short (int): 短期 EMA の期間。
        long (int): 長期 EMA の期間。
        tp_mult (float): 利確幅の ATR 倍率。

    Returns:
        Backtester: 新しく作った Backtester。
    """
    return Backtester(
        indicators=[MAIndicator("EMA", short, long), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(tp_mult),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
        init_cash=INIT_CASH,
    )


def lot_cost_table(data: dict[str, pd.DataFrame], init_cash: float) -> pd.DataFrame:
    """
    銘柄ごとに、1単元（実際の100株）の金額と、初期資金に対する割合を求めます。

    1単元が高い銘柄は、サイズが1単元に届かずエントリー注文が捨てられやすくなります。
    期間中の最小・最大を出すのは、株価が上がって途中から買えなくなる銘柄があるためです。

    Args:
        data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
        init_cash (float): 初期資金。

    Returns:
        pd.DataFrame: 銘柄をインデックスとする表。列は lot_cost_min / lot_cost_max / max_ratio。
    """
    rows = {}
    for ticker, df in data.items():
        raw_open = df["RawOpen"] if "RawOpen" in df.columns else df["Open"]
        lot_cost = raw_open * UNIT_SHARES
        rows[ticker] = {
            "lot_cost_min": lot_cost.min(),
            "lot_cost_max": lot_cost.max(),
            "max_ratio": lot_cost.max() / init_cash,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    setup_logging()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    universe = load_universe_file(universe_path(UNIVERSE_NAME))
    report.print_universe_note(universe, UNIVERSE_NAME)
    # 上場前のデータが壊れている銘柄は、ユニバースの data_from の日から使う（#70）
    data = load_universe(
        universe["ticker"], start="2010-01-01", data_from=data_from_map(universe)
    )
    mb = MultiBacktester(build_strategy, PARAM_GRID)

    # 0. 初期資金に対する1単元の金額（高い銘柄はエントリー注文が捨てられやすい）
    init_cash = build_strategy().init_cash
    lot_costs = lot_cost_table(data, init_cash)
    print("【1単元の金額（初期資金に対する割合）】")
    print(lot_costs)
    for ticker in lot_costs.index[lot_costs["max_ratio"] > 1]:
        logger.warning(f"[{ticker}] 期間中に1単元の金額が初期資金を超えています。")

    # 1. 銘柄 × パラメータごとに独立して実行
    # 業種・規模で集計できるように、ユニバースの属性を結合する。どのユニバースの結果かを CSV に残す
    summary = attach_attributes(mb.run_independent(data, start=EVAL_START), universe)
    summary.insert(0, "universe", UNIVERSE_NAME)
    summary.to_csv("multi_summary.csv", index=False)
    # スプレッドシートに出力する表（シート名の接頭辞を除いた名前 -> 表）
    sheet_tables: dict[str, pd.DataFrame] = {"Summary": summary}

    # 2. パラメータごとに銘柄横断で集計（多くの銘柄で安定しているかを見る）
    by_params = mb.aggregate(summary)
    by_params.insert(0, "universe", UNIVERSE_NAME)
    by_params.to_csv("multi_by_params.csv", index=False)
    sheet_tables["ByParams"] = by_params
    print("【パラメータごとの銘柄横断集計】")
    print(by_params.sort_values("sharpe_median", ascending=False))
    # 銘柄ごとの買い持ちとの比較（#8）。beat_bh_ratio は、買い持ちより CAGR が高かった銘柄の割合
    print("【パラメータごとの、銘柄ごとの買い持ちとの比較】")
    print(
        by_params[
            [*PARAM_GRID, "cagr_median", "bh_cagr_median", "excess_cagr_median", "beat_bh_ratio"]
        ].sort_values("excess_cagr_median", ascending=False)
    )

    # 2'. パラメータ × 規模区分・業種で集計（特定の規模・業種だけで効いていないかを見る）
    for col in ["size", "sector"]:
        by_attr = mb.aggregate(summary, by=[col])
        by_attr.insert(0, "universe", UNIVERSE_NAME)
        by_attr.to_csv(f"multi_by_{col}.csv", index=False)
        sheet_tables[f"By{col.capitalize()}"] = by_attr
    print("【パラメータ × 規模区分の集計（sharpe_median）】")
    print(
        mb.aggregate(summary, by=["size"])
        .pivot_table(index=list(PARAM_GRID), columns="size", values="sharpe_median", sort=False)
    )

    # 3. 資金共有モード（strategy01 と同じパラメータで、全銘柄を1つの口座で運用）
    portfolio = mb.run_portfolio(
        data, params={"short": 5, "long": 20, "tp_mult": 4}, start=EVAL_START
    )
    print("【資金共有モードの結果】")
    report.print_summary(portfolio)
    report.save_trades_csv(portfolio, "multi_portfolio_trades.csv")

    # 3'. 買い持ちとの比較（#8）。同じ評価期間・初期資金・取引コストで、
    # - ユニバースの等金額の買い持ち: 銘柄の選び方（生存者バイアスを含む）を除いた、戦略だけの差を見る
    # - TOPIX 連動 ETF の買い持ち: 市場全体の追い風を除いても勝っているかを見る
    reference = build_strategy()
    bh_params = {"start": EVAL_START, "init_cash": reference.init_cash, **reference.cost_params}
    benchmarks = {"ユニバース等金額 買い持ち": equal_weight_buy_and_hold(data, **bh_params)}
    index_df = load_universe([BENCHMARK_TICKER], start="2010-01-01").get(BENCHMARK_TICKER)
    if index_df is None:
        logger.warning(f"{BENCHMARK_TICKER} のデータが取得できなかったため、比較から外しました。")
    else:
        benchmarks[f"{BENCHMARK_TICKER} 買い持ち"] = buy_and_hold(index_df, **bh_params)
    comparison = report.print_benchmark_comparison(portfolio, benchmarks)
    comparison.to_csv("multi_benchmark.csv")
    sheet_tables["Benchmark"] = comparison.reset_index(names="metric")

    # 4. スプレッドシートへの出力（実行するたびに同じシートを上書きする）
    if not SAVE_TO_SHEET:
        pass
    elif not report.sheet_is_configured():
        logger.warning(
            "SPREADSHEET_ID が設定されていないため、スプレッドシートへの出力を飛ばしました（CSV は出力済み）。"
        )
    else:
        for name, table in sheet_tables.items():
            report.save_table_to_sheet(table, f"{SHEET_PREFIX}{name}")
        report.save_trades_to_sheet(portfolio, sheet_name=f"{SHEET_PREFIX}PortfolioTrades")
