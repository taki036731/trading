"""
買い持ちのベンチマーク（#8）。

戦略の成績が「売買のタイミングの力」なのか「相場の追い風」なのかを区別するために、
同じ評価期間・同じ取引コスト・同じ年率換算（lib.metrics.account_metrics）で、買い持ちの成績を計算します。

- buy_and_hold: 1銘柄の買い持ち（その銘柄自身、TOPIX 連動 ETF など）
- equal_weight_buy_and_hold: 複数銘柄を初期資金の等分で買って持ち続ける（ユニバース全体の買い持ち）

買い持ちは、評価期間の初日の始値（スリッページ込み）で現金の全額を買い、最終日の終値で評価します（最後に売らない）。
始値が 0 や NaN の行（売買のなかった日など）では買えないので、その場合は買える最初の行で買います。
価格は戦略と同じ調整後価格なので、配当込みのリターンになります。
100株単元はかけず、端株も買います（単元をかけると、値がさ株では使えない現金が残り、
リターンが銘柄の値動きでなく現金の比率で変わってしまうため）。
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from lib.backtester import DateLike, period_mask
from lib.metrics import ACCOUNT_METRIC_COLUMNS, account_metrics

# 市場全体のベンチマーク（TOPIX 連動 ETF）
BENCHMARK_TICKER = "1306.T"

# run_independent の結果に足す、銘柄ごとの買い持ちの列
BH_METRIC_COLUMNS = ["bh_total_return", "bh_cagr", "bh_max_drawdown", "bh_sharpe"]
# 戦略の CAGR − 買い持ちの CAGR
EXCESS_COLUMN = "excess_cagr"


def _eval_rows(df: pd.DataFrame, start: DateLike | None, end: DateLike | None) -> pd.DataFrame:
    """評価期間 [start, end] の行を返します（行がなければ ValueError）。"""
    return df.loc[period_mask(df.index, start, end)]


def _first_tradable(rows: pd.DataFrame) -> int:
    """
    始値と終値がどちらも正の、最初の行の位置を返します。

    yfinance のデータには、売買のなかった日などで始値が 0 や NaN の行があり、
    その行では vectorbt が注文を受け付けない（ValueError）ため、買うのはこの行からにします。

    Args:
        rows (pd.DataFrame): 評価期間の行。

    Returns:
        int: 買える最初の行の位置（0 始まり）。

    Raises:
        ValueError: 評価期間に買える行がない場合。
    """
    # NaN との比較は False になるので、NaN の行も買えない行になる
    tradable = ((rows["Open"] > 0) & (rows["Close"] > 0)).to_numpy()
    if not tradable.any():
        raise ValueError("エラー: 評価期間に、始値と終値が正の行がありません。")
    return int(np.argmax(tradable))


def _close_for_valuation(close: pd.DataFrame) -> pd.DataFrame:
    """
    時価評価に使う終値を返します。0 以下と NaN を前の値で埋めます（過去方向のみ）。

    終値が 0 や NaN の行があると、その日の評価額が 0 や NaN になり、最大ドローダウンなどが壊れるためです。

    Args:
        close (pd.DataFrame): 終値（列=銘柄）。

    Returns:
        pd.DataFrame: 埋めた終値。
    """
    return close.mask(close <= 0).ffill()


def buy_and_hold(
    df: pd.DataFrame,
    start: DateLike | None = None,
    end: DateLike | None = None,
    init_cash: float = 1_000_000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> vbt.Portfolio:
    """
    1銘柄の買い持ちをシミュレーションします。

    評価期間の初日の始値で現金の全額を買い、最終日の終値で評価します。
    始値か終値が 0・NaN の行では買えないので、その場合は買える最初の行で買います。

    Args:
        df (pd.DataFrame): Open / Close 列を持つ株価データ（調整後価格）。
        start (DateLike | None): 評価期間の開始日（その日を含む）。None の場合は df の先頭から。
        end (DateLike | None): 評価期間の終了日（その日を含む）。None の場合は df の末尾まで。
        init_cash (float): 初期資金。
        fees (float): 手数料（片道の割合）。買いの1回だけにかかります。
        slippage (float): スリッページ（片道の割合）。買いの1回だけにかかります。

    Returns:
        vbt.Portfolio: 評価期間の行だけの Portfolio。

    Raises:
        ValueError: 評価期間にデータがない場合。
    """
    rows = _eval_rows(df, start, end)
    size = pd.Series(np.nan, index=rows.index)
    # 現金で買える全額。始値が 0 や NaN の行では買えないので、買える最初の行で買う
    size.iloc[_first_tradable(rows)] = np.inf
    return vbt.Portfolio.from_orders(
        close=_close_for_valuation(rows[["Close"]])["Close"],
        price=rows["Open"],
        size=size,
        direction="longonly",
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq="D",
    )


def equal_weight_buy_and_hold(
    data: dict[str, pd.DataFrame],
    start: DateLike | None = None,
    end: DateLike | None = None,
    init_cash: float = 1_000_000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> vbt.Portfolio:
    """
    複数銘柄を初期資金の等分で買い、持ち続ける（リバランスなし）買い持ちをシミュレーションします。

    1つの口座（資金共有）で、各銘柄の評価期間の初日（始値か終値が 0・NaN なら、買える最初の行）の始値で
    init_cash / 銘柄数 ずつ買います。
    評価期間の途中で上場した銘柄の分は、上場日まで現金で持ちます。
    日付は全銘柄の和集合にそろえ、終値は時価評価のため前方埋めします（過去方向のみ）。

    Args:
        data (dict[str, pd.DataFrame]): ティッカーをキー、株価データを値とする辞書。
        start (DateLike | None): 評価期間の開始日（その日を含む）。
        end (DateLike | None): 評価期間の終了日（その日を含む）。
        init_cash (float): 口座全体の初期資金。
        fees (float): 手数料（片道の割合）。
        slippage (float): スリッページ（片道の割合）。

    Returns:
        vbt.Portfolio: 全銘柄を1グループにまとめた Portfolio。

    Raises:
        ValueError: data が空の場合、または評価期間にデータがない銘柄がある場合。
    """
    if not data:
        raise ValueError("エラー: 銘柄データが空です。")
    rows = {ticker: _eval_rows(df, start, end) for ticker, df in data.items()}
    close = _close_for_valuation(pd.DataFrame({t: r["Close"] for t, r in rows.items()}).sort_index())
    price = pd.DataFrame({t: r["Open"] for t, r in rows.items()}).reindex(close.index)

    # 各銘柄の買える最初の行だけに、買う金額を置く。vectorbt の size_type="value" はスリッページ前の価格で
    # 株数を決めるため、手数料とスリッページの分を差し引いて、各銘柄にちょうど等分の現金を使う
    per_ticker = init_cash / len(rows) / (1.0 + fees) / (1.0 + slippage)
    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for ticker, r in rows.items():
        size.loc[r.index[_first_tradable(r)], ticker] = per_ticker

    return vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=size,
        size_type="value",
        direction="longonly",
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        group_by=True,
        cash_sharing=True,
        freq="D",
    )


def benchmark_metrics(portfolio: vbt.Portfolio) -> dict[str, float]:
    """
    買い持ちの Portfolio から、戦略と同じ基準の口座単位の指標を計算します。

    Args:
        portfolio (vbt.Portfolio): buy_and_hold / equal_weight_buy_and_hold の戻り値。

    Returns:
        dict[str, float]: ACCOUNT_METRIC_COLUMNS をキーとする指標の辞書。
    """
    return account_metrics(portfolio)


def bh_columns(metrics: dict[str, float], strategy_cagr: float) -> dict[str, float]:
    """
    買い持ちの指標を、run_independent の結果に足す列（BH_METRIC_COLUMNS と EXCESS_COLUMN）にします。

    Args:
        metrics (dict[str, float]): benchmark_metrics の戻り値。
        strategy_cagr (float): 戦略の CAGR。

    Returns:
        dict[str, float]: BH_METRIC_COLUMNS と EXCESS_COLUMN をキーとする辞書。
    """
    return {
        "bh_total_return": metrics["total_return"],
        "bh_cagr": metrics["cagr"],
        "bh_max_drawdown": metrics["max_drawdown"],
        "bh_sharpe": metrics["sharpe"],
        EXCESS_COLUMN: strategy_cagr - metrics["cagr"],
    }


__all__ = [
    "ACCOUNT_METRIC_COLUMNS",
    "BENCHMARK_TICKER",
    "BH_METRIC_COLUMNS",
    "EXCESS_COLUMN",
    "benchmark_metrics",
    "bh_columns",
    "buy_and_hold",
    "equal_weight_buy_and_hold",
]
