import pandas_ta as ta  # noqa: F401

from lib import data_loader as dl
from lib import report
from lib import setup_logging
from lib.backtester import Backtester
from lib.benchmark import BENCHMARK_TICKER, buy_and_hold
from lib.multi_backtester import load_universe
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

# 評価期間の開始日。データはインジケーターのウォームアップのために、これより前から取得する（#38）
EVAL_START = "2016-06-25"

TICKER = "7203.T"

if __name__ == "__main__":
    setup_logging()
    df = dl.fetch_stock_data(TICKER, start="2010-01-01")

    # バックテスト実行
    backtester = Backtester(
        indicators=[MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries=[
            GoldenCrossSignal(),
            ATRConditionSignal(lower=1.5, upper=5),
        ],
        exits=[DeadCrossSignal()],
        tp=AtrTakeProfit(4),
        sl=AtrStopLoss(2),
        position_sizer=RiskPercentageSizer(0.02),
    )

    portfolio = backtester.run(df, start=EVAL_START)
    report.print_summary(portfolio)

    # 同じ評価期間・初期資金・取引コストで、その銘柄と TOPIX 連動 ETF の買い持ちと比べる（#8）
    bh_params = {"start": EVAL_START, "init_cash": backtester.init_cash, **backtester.cost_params}
    benchmarks = {f"{TICKER} 買い持ち": buy_and_hold(df, **bh_params)}
    # 取得に失敗しても（オフラインでキャッシュがない場合など）、比較から外すだけで止めない
    index_df = load_universe([BENCHMARK_TICKER], start="2010-01-01").get(BENCHMARK_TICKER)
    if index_df is None:
        print(f"{BENCHMARK_TICKER} のデータが取得できなかったため、比較から外しました。")
    else:
        benchmarks[f"{BENCHMARK_TICKER} 買い持ち"] = buy_and_hold(index_df, **bh_params)
    report.print_benchmark_comparison(portfolio, benchmarks)
    report.save_orders_csv(portfolio)
    # チャートには、run() が残すインジケーターの列を足した株価データとシグナルを使う
    assert backtester.df is not None
    report.save_chart_to_sheet(
        portfolio,
        backtester.df,
        backtester.entries_series,
        backtester.exits_series,
    )
    report.save_orders_to_sheet(portfolio)
    report.save_trades_to_sheet(portfolio)
    # report.show(portfolio)
    input("Press Enter to close ...")
