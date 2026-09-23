import pandas as pd
import vectorbt as vbt

from lib.strategies.exits import AbstractStopLoss, AbstractTakeProfit
from lib.strategies.indicators import AbstractIndicator
from lib.strategies.position_sizing import AbstractPositionSizer
from lib.strategies.signals import AbstractSignal


class Backtester:
    def __init__(
        self,
        indicators: list[AbstractIndicator],
        entries: list[AbstractSignal],
        exits: list[AbstractSignal],
        tp: AbstractTakeProfit,
        sl: AbstractStopLoss,
        position_sizer: AbstractPositionSizer,
        init_cash: float = 1000000.0,
    ):
        self.indicators = indicators
        self.entries = entries
        self.exits = exits
        self.tp = tp
        self.sl = sl
        self.position_sizer = position_sizer
        self.init_cash = init_cash
        self.portfolio: vbt.Portfolio | None = None

    def _generate_signals(
        self, df: pd.DataFrame, signals: list[AbstractSignal]
    ) -> pd.Series:
        if not signals:
            return pd.Series(False, index=df.index)
        results = [s.generate(df) for s in signals]
        return pd.concat(results, axis=1).all(axis=1)

    def run(self, df: pd.DataFrame) -> vbt.Portfolio:
        # ---------------------------------------------------------
        # 1. インジケーターの計算 (ベクトル演算で一括処理)
        # ---------------------------------------------------------
        for indicator in self.indicators:
            indicator.generate(df)

        # ---------------------------------------------------------
        # 2. 動的な利確・損切り幅の設定
        # ---------------------------------------------------------
        sl_pct = self.sl.generate(df)
        tp_pct = self.tp.generate(df)

        # ---------------------------------------------------------
        # 3. バックテストの実行
        # ---------------------------------------------------------
        # ポジションサイズの計算
        size = self.position_sizer.generate(df, sl_pct=sl_pct)

        self.portfolio = vbt.Portfolio.from_signals(
            close=df["Close"],
            entries=self._generate_signals(df, self.entries),
            exits=self._generate_signals(df, self.exits),
            sl_stop=sl_pct,  # 動的ストップロスの配列
            tp_stop=tp_pct,  # 動的テイクプロフィットの配列
            size=size,
            size_type=self.position_sizer.size_type,
            size_granularity=100,  # 日本株固定ルール（100株単位）
            min_size=100,  # 100株未満は注文しない
            init_cash=self.init_cash,  # 初期資金
            fees=0.0,  # 手数料設定（必要に応じて）
            freq="D",  # 日足データ
        )
        return self.portfolio

    def display_stats(self):
        if self.portfolio is None:
            print("バックテストが実行されていません。先に run() を呼び出してください。")
            return

        print("-" * 40)
        print("【バックテスト結果サマリ】")
        # 勝率、プロフィットファクター、ドローダウンなどが一括で計算されます
        print(self.portfolio.stats())
        print("-" * 40)

    def save_orders_to_csv(self, filename: str = "vectorbt_orders.csv"):
        if self.portfolio is None:
            print("バックテストが実行されていません。")
            return
        # 取引履歴の詳細をCSVとして出力
        self.portfolio.orders.records_readable.to_csv(filename)  # type: ignore
        print(f"注文履歴を {filename} に保存しました。")
