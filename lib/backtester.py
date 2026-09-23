import logging

import pandas as pd
import vectorbt as vbt

import lib.data_saver as ds
from lib.strategies.exits import AbstractStopLoss, AbstractTakeProfit
from lib.strategies.indicators import AbstractIndicator
from lib.strategies.position_sizing import AbstractPositionSizer
from lib.strategies.signals import AbstractSignal

logger = logging.getLogger(__name__)


class Backtester:
    """
    売買戦略のバックテストを実行・管理するクラス。

    インジケーターの計算、売買シグナルの生成、ポジションサイジング、
    および vectorbt を使用したシミュレーションの実行を担います。
    """

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
        """
        Backtester を初期化します。

        Args:
            indicators (list[AbstractIndicator]): 計算するインジケーターのリスト。
            entries (list[AbstractSignal]): エントリー（買い）条件シグナルのリスト。
            exits (list[AbstractSignal]): エグジット（売り）条件シグナルのリスト。
            tp (AbstractTakeProfit): 利確ロジックの実装。
            sl (AbstractStopLoss): 損切りロジックの実装。
            position_sizer (AbstractPositionSizer): ポジションサイズ決定ロジック。
            init_cash (float): 初期資金。デフォルトは 1,000,000.0。
        """
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
        """
        複数のシグナル条件を統合し、すべてを満たす箇所を判定します。

        Args:
            df (pd.DataFrame): 株価および指標データ。
            signals (list[AbstractSignal]): 判定するシグナルオブジェクトのリスト。

        Returns:
            pd.Series: すべての条件を満たす場合に True となるブール値Series。
        """
        if not signals:
            return pd.Series(False, index=df.index)
        results = [s.generate(df) for s in signals]
        return pd.concat(results, axis=1).all(axis=1)

    def run(self, df: pd.DataFrame) -> vbt.Portfolio:
        """
        バックテストを実行します。

        1. インジケーターの計算
        2. 動的な利確・損切り幅の算出
        3. ポジションサイズの決定
        4. vectorbt によるシミュレーション実行

        Args:
            df (pd.DataFrame): バックテストに使用する株価データ。

        Returns:
            vbt.Portfolio: バックテスト結果を含む vectorbt の Portfolio オブジェクト。
        """
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
        """
        バックテストの結果サマリを表示します。
        """
        if self.portfolio is None:
            logger.warning(
                "バックテストが実行されていません。先に run() を呼び出してください。"
            )
            return

        print("-" * 40)
        print("【バックテスト結果サマリ】")
        # 勝率、プロフィットファクター、ドローダウンなどが一括で計算されます
        print(self.portfolio.stats())
        print("-" * 40)

    def save_orders_to_csv(self, filename: str = "vectorbt_orders.csv"):
        """
        取引履歴をCSV形式で保存します。

        Args:
            filename (str): 保存するファイル名。デフォルトは 'vectorbt_orders.csv'。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return
        # 取引履歴の詳細をCSVとして出力
        self.portfolio.orders.records_readable.to_csv(filename)  # type: ignore
        logger.info(f"注文履歴を {filename} に保存しました。")

    def save_chart_to_sheet(self):
        """
        バックテストの取引履歴をスプレッドシートに保存します。

        `lib.data_saver` を使用して、'Chart' シートに履歴データを出力します。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return

        history_df = self.portfolio.orders.records_readable.rename(  # type: ignore
            columns={"Timestamp": "Date"}
        )
        history_df = history_df.fillna("")
        ds.write_df_to_sheet("Chart", history_df)

        # 資産推移やドローダウンのチャートも1行で描画可能です
        # portfolio.plot().show()

    def show(self):
        """
        バックテストの結果をチャートで表示します。
        """
        if self.portfolio is None:
            logger.warning(
                "バックテストが実行されていません。先に run() を呼び出してください。"
            )
            return

        self.portfolio.plot().show()  # type: ignore
