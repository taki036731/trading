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
        self.df: pd.DataFrame | None = None
        self.entries_series: pd.Series | None = None
        self.exits_series: pd.Series | None = None

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
        # 前日の終値を基に、翌日の損切り・利確%を決める。
        # そのため、StopLoss / TakeProfit のパーセンテージを「翌日約定用」にシフトする
        sl_pct = self.sl.generate(df).shift(1).fillna(0.0)
        tp_pct = self.tp.generate(df).shift(1).fillna(0.0)

        # ---------------------------------------------------------
        # 3. エントリーとエグジットのシグナルを生成
        # ---------------------------------------------------------
        entries = self._generate_signals(df, self.entries)
        exits = self._generate_signals(df, self.exits)

        # 状態を保持 (参照のみ)
        self.df = df
        self.entries_series = entries
        self.exits_series = exits

        # ---------------------------------------------------------
        # 4. ポジションサイズの計算
        # ---------------------------------------------------------
        # サイズ計算にもDay TのSL%を渡す(shift(1)した配列を渡す)
        size = self.position_sizer.generate(df, sl_pct=sl_pct)

        # ---------------------------------------------------------
        # 5. バックテストの実行
        # ---------------------------------------------------------
        self.portfolio = vbt.Portfolio.from_signals(
            close=df["Close"],
            # SL/TP をザラ場の値動きで判定するために OHLC を渡す。
            # 渡さないと終値だけで判定され、約定価格も終値になってしまう
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            entries=entries,
            exits=exits,
            price=df["Open"],
            direction="longonly",
            sl_stop=sl_pct,  # 動的ストップロスの配列
            tp_stop=tp_pct,  # 動的テイクプロフィットの配列
            # SL/TP ラインの基準を約定価格(始値)にする。
            # 既定値はエントリー日の終値で、想定したラインからずれる
            stop_entry_price="fillprice",
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
        バックテストの結果サマリと最新の注文履歴を表示します。
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
        print("【最新の注文履歴】")
        # 最新5件を表示
        print(self.portfolio.orders.records_readable.tail(5))  # type: ignore
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
        価格、指標、シグナル、売買履歴を結合した一時的なテーブルを作成し、
        スプレッドシートの 'Chart' シートに保存します。
        """
        if (
            self.portfolio is None
            or self.df is None
            or self.entries_series is None
            or self.exits_series is None
        ):
            logger.warning("バックテストが実行されていません。")
            return

        # 1. 注文履歴を日付で集約
        orders = self.portfolio.orders.records_readable.copy()  # type: ignore
        orders["Timestamp"] = pd.to_datetime(orders["Timestamp"])

        # 売買代金 (Value) を算出
        orders["Value"] = orders["Size"] * orders["Price"]

        # 同じ日に複数注文がある可能性を考慮して集約
        order_summary = orders.groupby("Timestamp").agg(
            {
                "Size": "sum",
                "Value": "sum",
                "Side": lambda x: ",".join(x.astype(str).unique()),
            }
        )

        # 加重平均価格を算出 (Total Value / Total Size)
        order_summary["Price"] = order_summary["Value"] / order_summary["Size"]

        # カラム名の整理
        order_summary = order_summary.rename(
            columns={
                "Size": "exec_size",
                "Price": "exec_price",
                "Value": "exec_value",
                "Side": "exec_side",
            }
        )[["exec_size", "exec_price", "exec_value", "exec_side"]]

        # 2. データをこの瞬間だけ結合
        export_df = pd.concat(
            [
                self.df,
                self.entries_series.rename("entry_signal"),
                self.exits_series.rename("exit_signal"),
                order_summary,
            ],
            axis=1,
        )

        # 3. スプレッドシートに書き込み
        ds.write_df_to_sheet("Chart", export_df.fillna(""), include_index=True)
        logger.info("売買履歴を含む時系列データを 'Chart' シートに保存しました。")

    def save_orders_to_sheet(self, sheet_name: str = "Orders"):
        """
        詳細な注文履歴をスプレッドシートに保存します。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return

        df = self.portfolio.orders.records_readable  # type: ignore
        ds.write_df_to_sheet(sheet_name, df, include_index=False)
        logger.info(f"注文履歴を '{sheet_name}' シートに保存しました。")

    def save_trades_to_sheet(self, sheet_name: str = "Trades"):
        """
        取引履歴（エントリーからエグジットまでの一連の結果）をスプレッドシートに保存します。
        """
        if self.portfolio is None:
            logger.warning("バックテストが実行されていません。")
            return

        df = self.portfolio.trades.records_readable  # type: ignore
        ds.write_df_to_sheet(sheet_name, df, include_index=False)
        logger.info(f"取引履歴を '{sheet_name}' シートに保存しました。")

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
