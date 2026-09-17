import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import vectorbt as vbt

from lib import data_loader as dl
from lib import setup_logging
from lib.strategies.exits import (
    AbstractStopLoss,
    AbstractTakeProfit,
    AtrStopLoss,
    AtrTakeProfit,
)
from lib.strategies.indicators import AbstractIndicator, ATRIndicator, MAIndicator
from lib.strategies.position_sizing import AbstractPositionSizer, RiskPercentageSizer
from lib.strategies.signals import (
    AbstractSignal,
    ATRConditionSignal,
    DeadCrossSignal,
    GoldenCrossSignal,
    InDateRangeSignal,
)


def generate_signals(df: pd.DataFrame, signals: list[AbstractSignal]) -> list:
    retval = []
    for s in signals:
        r = s.generate(df)
        retval.append(r)
    return retval


def run_vectorbt_backtest(
    indicators: list[AbstractIndicator],
    entries: list[AbstractSignal],
    exits: list[AbstractSignal],
    tp: AbstractTakeProfit,
    sl: AbstractStopLoss,
    position_sizer: AbstractPositionSizer,
    df: pd.DataFrame,
):
    # ---------------------------------------------------------
    # 1. インジケーターの計算 (ベクトル演算で一括処理)
    # ---------------------------------------------------------
    for indicator in indicators:
        indicator.generate(df)

    # ---------------------------------------------------------
    # 3. 動的な利確・損切り幅の設定
    # ---------------------------------------------------------
    # vectorbtの sl_stop / tp_stop には「現在の価格に対するパーセンテージ」を配列で渡せます
    sl_pct = sl.generate(df)
    tp_pct = tp.generate(df)

    # ---------------------------------------------------------
    # 4. バックテストの実行 (内部はC言語レベルで高速処理)
    # ---------------------------------------------------------
    # ポジションサイズの計算
    size = position_sizer.generate(df, sl_pct=sl_pct)

    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=pd.concat(generate_signals(df, entries), axis=1).all(axis=1),
        exits=pd.concat(generate_signals(df, exits), axis=1).all(axis=1),
        sl_stop=sl_pct,  # 動的ストップロスの配列
        tp_stop=tp_pct,  # 動的テイクプロフィットの配列
        size=size,
        size_type=position_sizer.size_type,
        size_granularity=100,  # 日本株固定ルール（100株単位）
        min_size=100,  # 100株未満は注文しない
        init_cash=1000000.0,  # 初期資金
        fees=0.0,  # 手数料設定（必要に応じて）
        freq="D",  # 日足データ
    )

    # ---------------------------------------------------------
    # 5. 分析結果の出力
    # ---------------------------------------------------------
    print("-" * 40)
    print("【バックテスト結果サマリ】")
    # 勝率、プロフィットファクター、ドローダウンなどが一括で計算されます
    print(portfolio.stats())
    print("-" * 40)

    # 取引履歴の詳細をCSVとして出力することも容易です
    portfolio.orders.records_readable.to_csv("vectorbt_orders.csv")  # type: ignore
    # ds.write_df_to_sheet("Order", portfolio.orders.records_readable)  # type: ignore

    # 指定期間（start_dateからend_date）でフィルタリングしてCSV出力
    # df[(df.index >= start_date) & (df.index <= end_date)].to_csv("chart.csv")
    # ds.write_df_to_sheet("Chart", df[(df.index >= start_date) & (df.index <= end_date)])

    ## Google Spread Sheetに出力する場合
    # history_df = pd.merge(
    #     df[(df.index >= start_date) & (df.index <= end_date)].reset_index(),
    #     portfolio.orders.records_readable.rename(columns={"Timestamp": "Date"}),  # type: ignore
    #     on="Date",
    #     how="left",
    # )
    # history_df = history_df.fillna("")
    # ds.write_df_to_sheet("Chart", history_df)

    # 資産推移やドローダウンのチャートも1行で描画可能です
    # portfolio.plot().show()


if __name__ == "__main__":
    setup_logging()
    # --- 検証用のダミーデータ生成 ---
    # 実際の運用では、作成済みの data_loader.py から取得したDataFrameを渡します
    print("ダミーデータを生成しています...")
    np.random.seed(42)
    days = 1500
    price_walk = 1500 + np.random.randn(days).cumsum() * 10

    # OHLCデータフレームの構築
    dummy_df = pd.DataFrame(
        {
            "open": price_walk + np.random.randn(days) * 2,
            "high": price_walk + np.random.rand(days) * 10,
            "low": price_walk - np.random.rand(days) * 10,
            "Close": price_walk,
        },
        index=pd.date_range("2020-01-01", periods=days),
    )
    df = dl.fetch_stock_data("7203.T", start="2010-01-01")

    # バックテスト実行
    entries = [
        GoldenCrossSignal(),
        InDateRangeSignal(
            datetime.datetime(2016, 6, 25), datetime.datetime(2030, 12, 31)
        ),
        ATRConditionSignal(lower=1.5, upper=5),
    ]

    run_vectorbt_backtest(
        [MAIndicator("EMA", 5, 20), ATRIndicator(15)],
        entries,
        [DeadCrossSignal()],
        AtrTakeProfit(4),
        AtrStopLoss(2),
        RiskPercentageSizer(0.02),
        df,
    )
