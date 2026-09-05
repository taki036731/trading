import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import vectorbt as vbt

import lib.strategies.indicator.abstract_indicator as ai
import lib.strategies.indicator.atr_indicator as atr
import lib.strategies.indicator.ma_indicator as ma
from lib import data_loader as dl
from lib import setup_logging


def run_vectorbt_backtest(indicators: list[ai.AbstractIndicator], df: pd.DataFrame):
    # ---------------------------------------------------------
    # 1. インジケーターの計算 (ベクトル演算で一括処理)
    # ---------------------------------------------------------
    for indicator in indicators:
        indicator.generate(df)

    # ---------------------------------------------------------
    # 2. シグナル生成 (論理演算で一括判定)
    # ---------------------------------------------------------
    # ゴールデンクロス・デッドクロスの判定 (shift(1)で前日と比較)
    golden_cross = (df["short_ma"] > df["long_ma"]) & (
        df["short_ma"].shift(1) <= df["long_ma"].shift(1)
    )
    dead_cross = (df["short_ma"] < df["long_ma"]) & (
        df["short_ma"].shift(1) >= df["long_ma"].shift(1)
    )
    start_date = datetime.datetime(2016, 6, 25)
    end_date = datetime.datetime(2030, 12, 31)
    in_date_range = (df.index >= start_date) & (df.index <= end_date)

    # ATR比率のフィルター (1.5% 〜 5.0%)
    df["atr_ratio"] = (df["atr"] / df["Close"]) * 100
    atr_condition = (df["atr_ratio"] >= 1.5) & (df["atr_ratio"] <= 5.0)

    # 最終的なエントリ・エグジットフラグ（True/Falseの配列）
    entries = golden_cross & atr_condition & in_date_range
    exits = dead_cross

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
        entries=entries,
        exits=exits,
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
    run_vectorbt_backtest(indicators, df)
