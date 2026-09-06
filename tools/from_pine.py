import csv
import datetime
from typing import Any

import backtrader as bt

from lib import data_loader as dl
from lib import setup_logging


class OsamuStrategy(bt.Strategy):
    """
    Pine Script (TradingView) から移植された、EMAクロスオーバーとATRを用いた戦略。
    """

    data: Any
    broker: Any
    position: Any

    params = (
        ("short_length", 5),  # (名前, デフォルト値) という形式のタプル
        ("long_length", 20),
        ("atr_length", 15),
        ("min_atr_ratio", 1.5),
        ("max_atr_ratio", 5.0),
        ("risk_rate", 2.0),
        ("start_date", datetime.datetime(2016, 6, 25)),
        ("end_date", datetime.datetime(2030, 12, 31)),
    )

    def __init__(self):
        """
        インジケーターの初期化とシグナルの設定。
        """
        # ... existing code ...
        self.daily_sell_qty = 0  # 当日の売り約定数を一時保存

    # 注文のステータスが変化したときに自動で呼び出されるメソッド
    def notify_order(self, order):
        """
        注文ステータスの変更通知。

        Args:
            order: 注文オブジェクト。
        """
        # 注文が「完了（約定）」した場合のみ処理
        if order.status in [order.Completed]:
            if order.isbuy():
                self.daily_buy_qty += order.executed.size
            elif order.issell():
                self.daily_sell_qty += order.executed.size

    def next(self):
        """
        各バー（日次データ）ごとの売買判定と執行。
        """
        current_date = self.data.datetime.datetime(0)
        # ... existing code ...
        self.daily_sell_qty = 0

    # バックテスト終了時に自動で呼び出されるメソッド
    def stop(self):
        """
        バックテスト終了時の処理。ログをCSVに出力します。
        """
        # リストに貯めたデータをCSVファイルとして書き出し
        with open("backtest_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # ヘッダー行を書き込み
            writer.writerow(
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Short_EMA",
                    "Long_EMA",
                    "ATR",
                    "Buy_Qty",
                    "Sell_Qty",
                ]
            )
            # データを一括で書き込み
            writer.writerows(self.log_data)
        print("CSVファイルの出力が完了しました（backtest_log.csv）")


if __name__ == "__main__":
    setup_logging()
    cerebro = bt.Cerebro()

    # データの取得（例としてAppleの株価を使用。開始・終了日時を指定）
    print("データをダウンロードしています...")
    df = dl.fetch_stock_data("7203.T", start="2010-01-01")
    data = bt.feeds.PandasData(dataname=df)  # type: ignore
    cerebro.adddata(data)

    # ストラテジーの追加
    cerebro.addstrategy(OsamuStrategy)

    # 初期資金の設定（1,000,000）
    cerebro.broker.setcash(1000000.0)

    # --- アナライザーの追加 ---
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")

    print(f"初期資金: {cerebro.broker.getvalue():,.2f}")

    # バックテストを実行し、結果を変数に格納します
    results = cerebro.run()

    # 実行されたストラテジーのインスタンス（1つ目）を取得します
    strat = results[0]

    # --- 分析結果の抽出と計算 ---
    dd_info = strat.analyzers.drawdown.get_analysis()
    trade_info = strat.analyzers.trade.get_analysis()

    print("-" * 40)
    print("【バックテスト結果（TradingView比較用）】")

    # 1. 総損益
    final_value = cerebro.broker.getvalue()
    net_profit = final_value - 1000000.0
    print(f"最終資金: {final_value:,.2f}")
    print(f"総損益: {net_profit:,.2f}")

    # 2. 最大ドローダウン（パーセンテージ）
    max_dd = dd_info.get("max", {}).get("drawdown", 0.0)
    print(f"最大ドローダウン: {max_dd:.2f} %")

    # 3. トレード数と勝ち数
    # 取引が1度も発生しなかった場合のエラーを防ぐため .get() を使用します
    total_trades = trade_info.get("total", {}).get("closed", 0)
    won_trades = trade_info.get("won", {}).get("total", 0)
    print(f"総トレード数: {total_trades}")
    print(f"勝ちトレード数: {won_trades}")

    # 4. プロフィットファクター (総利益 ÷ 総損失)
    # Backtraderには直接PFを出力する項目がないため、総利益と総損失から算出します
    gross_profit = trade_info.get("won", {}).get("pnl", {}).get("total", 0.0)
    gross_loss = trade_info.get("lost", {}).get("pnl", {}).get("total", 0.0)

    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)
        print(f"プロフィットファクター: {profit_factor:.3f}")
    else:
        print("プロフィットファクター: 算出不能 (損失トレードなし、または取引なし)")

    print("-" * 40)

    # チャートの描画
    cerebro.plot(style="candlestick")
