import datetime
import math

import backtrader as bt

from lib import data_loader as dl
from lib import setup_logging


class OsamuStrategy(bt.Strategy):
    # --- 1. パラメータ入力 ---
    # backtrader独自の仕様で、クラス変数 params に「タプルのタプル」を代入しています。
    # Pythonの標準的な文法としては、複数の要素を持つ入れ子状のイミュータブル（変更不可）な集合です。
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
        # --- 2. インジケーターの計算 ---
        self.short_ma = bt.indicators.ExponentialMovingAverage(
            self.data.close,  # type: ignore
            period=self.params.short_length,  # type: ignore
        )
        self.long_ma = bt.indicators.ExponentialMovingAverage(
            self.data.close,  # type: ignore
            period=self.params.long_length,  # type: ignore
        )
        self.atr = bt.indicators.AverageTrueRange(
            self.data,  # type: ignore
            period=self.params.atr_length,  # type: ignore
        )

        # ゴールデンクロス・デッドクロスの判定 (1: ゴールデン, -1: デッド, 0: クロスなし)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)  # type: ignore

        # 注文状態を管理
        self.order = None

    def next(self):
        # 進行中の注文があれば、重複エントリーを防ぐために何もしない
        if self.order:
            return

        # 現在のローソク足の日時を取得
        current_date = self.data.datetime.datetime(0)

        # inDateRange (指定期間内かどうかの判定)
        in_date_range = self.params.start_date <= current_date <= self.params.end_date  # type: ignore

        # ATR比率の計算 (%)
        atr_ratio = (self.atr[0] / self.data.close[0]) * 100

        # --- 4. 論理判定（フィルター）と 6. 注文の執行 ---
        # ポジションを持っていない場合（strategy.position_size == 0 に相当）
        if not self.position:
            atr_condition = (
                self.params.min_atr_ratio <= atr_ratio <= self.params.max_atr_ratio  # type: ignore
            )
            buy_signal = (self.crossover[0] == 1) and atr_condition and in_date_range

            if buy_signal:
                # --- 5. 資金管理アルゴリズム（2%ルール） ---
                current_equity = (
                    self.broker.getvalue()
                )  # 現在の総資産 (strategy.equity)
                max_loss_amount = current_equity * (self.params.risk_rate / 100.0)  # type: ignore
                risk_per_share = self.atr[0] * 2.0

                # ゼロ割りエラーを防止
                if risk_per_share > 0:
                    position_size = math.floor(max_loss_amount / risk_per_share)

                    if position_size > 0:
                        # ストップ価格と利益確定価格の計算
                        stop_loss_price = self.data.close[0] - (self.atr[0] * 2.0)
                        take_profit_price = self.data.close[0] + (self.atr[0] * 4.0)

                        # ブラケット注文 (Pineの strategy.exit に相当)
                        # エントリー注文の親に対し、利確・損切りの子注文を同時に紐づけます
                        entry_order = self.buy(size=position_size, transmit=False)
                        self.sell(
                            size=position_size,
                            price=take_profit_price,
                            exectype=bt.Order.Limit,
                            parent=entry_order,
                            transmit=False,
                        )
                        self.sell(
                            size=position_size,
                            price=stop_loss_price,
                            exectype=bt.Order.Stop,
                            parent=entry_order,
                            transmit=True,
                        )

        # ポジションを持っている場合
        else:
            sell_signal = self.crossover[0] == -1
            if sell_signal:
                # デッドクロスが発生した場合、保有中のポジションをすべて決済 (strategy.close に相当)
                self.close()


# --- バックテストの実行環境設定 ---
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
