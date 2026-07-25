import math

import backtrader as bt

from lib import data_loader as dl
from lib import setup_logging


class OsamuStrategy(bt.Strategy):
    # --- 1. パラメータ入力 ---
    # backtrader独自の仕様で、クラス変数 params に「タプルのタプル」を代入しています。
    # Pythonの標準的な文法としては、複数の要素を持つ入れ子状のイミュータブル（変更不可）な集合です。
    params = (
        ("short_length", 25),  # (名前, デフォルト値) という形式のタプル
        ("long_length", 75),
        ("atr_length", 14),
        ("min_atr_ratio", 1.5),
        ("max_atr_ratio", 5.0),
        ("risk_rate", 2.0),
    )

    def __init__(self):
        # --- 2. インジケーターの計算 ---
        self.short_ma = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.short_length
        )
        self.long_ma = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.long_length
        )
        self.atr = bt.indicators.AverageTrueRange(
            self.data, period=self.params.atr_length
        )

        # ゴールデンクロス・デッドクロスの判定 (1: ゴールデン, -1: デッド, 0: クロスなし)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

        # 注文状態を管理
        self.order = None

    def next(self):
        # 進行中の注文があれば、重複エントリーを防ぐために何もしない
        if self.order:
            return

        # ATR比率の計算 (%)
        atr_ratio = (self.atr[0] / self.data.close[0]) * 100

        # --- 4. 論理判定（フィルター）と 6. 注文の執行 ---
        # ポジションを持っていない場合（strategy.position_size == 0 に相当）
        if not self.position:
            atr_condition = (
                self.params.min_atr_ratio <= atr_ratio <= self.params.max_atr_ratio
            )
            buy_signal = (self.crossover[0] == 1) and atr_condition

            if buy_signal:
                # --- 5. 資金管理アルゴリズム（2%ルール） ---
                current_equity = (
                    self.broker.getvalue()
                )  # 現在の総資産 (strategy.equity)
                max_loss_amount = current_equity * (self.params.risk_rate / 100.0)
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
    df = dl.fetch_stock_data("AAPL", start="2015-01-01", end="2030-12-31")
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # ストラテジーの追加
    cerebro.addstrategy(OsamuStrategy)

    # 初期資金の設定（1,000,000）
    cerebro.broker.setcash(1000000.0)

    print(f"初期資金: {cerebro.broker.getvalue():.2f}")

    # バックテストを実行
    cerebro.run()

    print(f"最終資金: {cerebro.broker.getvalue():.2f}")

    # チャートの描画（Jupyter Notebook環境以外では別ウィンドウで開きます）
    cerebro.plot(style="candlestick")
