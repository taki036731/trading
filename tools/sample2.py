import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import vectorbt as vbt

from lib import data_loader as dl
from lib import setup_logging


def run_vectorbt_backtest(df: pd.DataFrame):
    # ---------------------------------------------------------
    # 1. インジケーターの計算 (ベクトル演算で一括処理)
    # ---------------------------------------------------------
    df["short_ma"] = df.ta.ema(length=5)
    df["long_ma"] = df.ta.ema(length=20)
    df["atr"] = df.ta.atr(length=15)

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

    # ATR比率のフィルター (1.5% 〜 5.0%)
    df["atr_ratio"] = (df["atr"] / df["Close"]) * 100
    atr_condition = (df["atr_ratio"] >= 1.5) & (df["atr_ratio"] <= 5.0)

    # 最終的なエントリ・エグジットフラグ（True/Falseの配列）
    entries = golden_cross & atr_condition
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
    # portfolio.orders.records_readable.to_csv("vectorbt_orders.csv")

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
    run_vectorbt_backtest(df)
