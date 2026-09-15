import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import vectorbt as vbt

import lib.strategies.indicators.abstract_indicator as ai
import lib.strategies.indicators.atr_indicator as atr
import lib.strategies.indicators.ma_indicator as ma
import lib.strategies.signals.abstract_signal as abs
import lib.strategies.signals.atr_condition_signal as ats
import lib.strategies.signals.dead_cross_signal as dcs
import lib.strategies.signals.golden_cross_signal as gcs
import lib.strategies.signals.in_date_range_signal as drs
from lib import data_loader as dl
from lib import setup_logging


def generate_signals(df: pd.DataFrame, signals: list[abs.AbstractSignal]) -> list:
    retval = []
    for s in signals:
        r = s.generate(df)
        retval.append(r)
    return retval


def run_vectorbt_backtest(
    indicators: list[ai.AbstractIndicator],
    entries: list[abs.AbstractSignal],
    exits: list[abs.AbstractSignal],
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
    sl_pct = (df["atr"] * 2.0) / df["Close"]  # 損切り: ATRの2倍
    tp_pct = (df["atr"] * 4.0) / df["Close"]  # 利確: ATRの4倍

    # ---------------------------------------------------------
    # 4. バックテストの実行 (内部はC言語レベルで高速処理)
    # ---------------------------------------------------------
    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=pd.concat(generate_signals(df, entries), axis=1).all(axis=1),
        exits=pd.concat(generate_signals(df, exits), axis=1).all(axis=1),
        sl_stop=sl_pct,  # 動的ストップロスの配列
        tp_stop=tp_pct,  # 動的テイクプロフィットの配列
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
    indicators = [ma.MAIndicator("EMA", 5, 20), atr.ATRIndicator(15)]
    entries = [
        gcs.GoldenCrossSignal(),
        drs.InDateRangeSignal(
            datetime.datetime(2016, 6, 25), datetime.datetime(2030, 12, 31)
        ),
        ats.ATRConditionSignal(1.5, 5),
    ]
    exits: list[abs.AbstractSignal] = [dcs.DeadCrossSignal()]
    run_vectorbt_backtest(indicators, entries, exits, df)
