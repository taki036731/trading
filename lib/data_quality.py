"""
株価データの検査（#70）。

取得元（yfinance など）のデータには、明らかな誤りが入っていることがあります。
誤りの行がそのまま残ると、最大ドローダウンやシャープレシオ、戦略のシグナルが壊れるため、
fetch_stock_data が返す前に、ここで検査して取り除きます。

- 終値が 0 以下の行: 価格が 0 以下になることはないので、誤りとして取り除きます
  （例: 8919.T の上場前のデータに、価格がマイナスの日がある）
- すぐに戻る急変: 終値が1日で SPIKE_RATIO 倍以上・分の1以下に動き、MAX_SPIKE_ROWS 行以内に
  元の水準（REVERT_TOLERANCE 以内）に戻った場合、その間の行を誤りとして取り除きます
  （例: 1306.T の 2026-03-30・03-31 だけ、価格が本来の 1/10 になっている）
- 戻らない急変: 本当の値動き（や、取得元の分割の調整漏れ）の可能性があり、どちらか判断できないので、
  警告だけにして残します

誤りの行は、前の日の値で埋めずに取り除きます。埋めると、実際にはなかった「値動きのない日」ができ、
その日に約定・判定したことになってしまうためです（取り除けば、その日はデータのない日として扱われます）。

すぐに戻る急変は、戻ったこと（その行より後のデータ）を見て判定します。これは価格の誤りを直すためのもので、
シグナルには使いません。データの最後の行の急変は、まだ戻るか分からないので、警告だけにします。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 1日の終値の変化が、この倍率以上（または分の1以下）なら急変とみなす
SPIKE_RATIO = 2.0
# 急変から元の水準に戻るまでの行数の上限（これより長く続く変化は、戻らない急変として扱う）
MAX_SPIKE_ROWS = 5
# 「元の水準に戻った」とみなす、急変の前の終値との差（割合）
REVERT_TOLERANCE = 0.25
# 警告に出す日付の数の上限
MAX_DATES_IN_LOG = 10


@dataclass
class QualityReport:
    """
    データの検査の結果です。

    Attributes:
        non_positive (list[pd.Timestamp]): 終値が 0 以下で、取り除いた行の日付。
        reverting_spikes (list[pd.Timestamp]): すぐに戻る急変で、取り除いた行の日付。
        jumps (list[pd.Timestamp]): 戻らない急変の日付（残した行）。
    """

    non_positive: list[pd.Timestamp] = field(default_factory=list)
    reverting_spikes: list[pd.Timestamp] = field(default_factory=list)
    jumps: list[pd.Timestamp] = field(default_factory=list)

    @property
    def dropped(self) -> list[pd.Timestamp]:
        """取り除いた行の日付（日付順）。"""
        return sorted(self.non_positive + self.reverting_spikes)


def _is_near(value: float, reference: float, tolerance: float) -> bool:
    """value が reference の ±tolerance（比率の対数で対称）以内なら True。"""
    return abs(np.log(value / reference)) <= np.log(1.0 + tolerance)


def _is_spike(value: float, reference: float, ratio: float) -> bool:
    """value が reference の ratio 倍以上、または ratio 分の1以下なら True。"""
    change = value / reference
    return change >= ratio or change <= 1.0 / ratio


def find_price_errors(
    df: pd.DataFrame,
    ratio: float = SPIKE_RATIO,
    max_rows: int = MAX_SPIKE_ROWS,
    tolerance: float = REVERT_TOLERANCE,
) -> QualityReport:
    """
    終値から、誤りの行（0 以下・すぐに戻る急変）と、戻らない急変を探します。

    比べる相手は、誤りでない直前の行の終値です（急変の途中の行とは比べない）。
    終値が NaN の行は飛ばします。

    Args:
        df (pd.DataFrame): Close 列を持つ株価データ（日付順）。
        ratio (float): 急変とみなす1日の変化の倍率。
        max_rows (int): 急変から元の水準に戻るまでの行数の上限。
        tolerance (float): 元の水準に戻ったとみなす、急変の前の終値との差（割合）。

    Returns:
        QualityReport: 検査の結果。
    """
    result = QualityReport()
    close = df["Close"]
    result.non_positive = list(close.index[close <= 0])

    valid = close[close > 0]
    values = valid.to_numpy(dtype=float)
    dates = valid.index
    if len(values) == 0:
        return result

    reference = values[0]
    i = 1
    while i < len(values):
        if not _is_spike(values[i], reference, ratio):
            reference = values[i]
            i += 1
            continue
        # 急変の後、max_rows 行以内に元の水準へ戻る最初の行を探す
        end = min(i + max_rows, len(values) - 1)
        back = next(
            (j for j in range(i + 1, end + 1) if _is_near(values[j], reference, tolerance)),
            None,
        )
        if back is None:
            result.jumps.append(dates[i])
            reference = values[i]
            i += 1
        else:
            result.reverting_spikes.extend(dates[i:back])
            reference = values[back]
            i = back + 1
    return result


def _format_dates(dates: list[pd.Timestamp]) -> str:
    """警告に出す日付の文字列（多い場合は先頭だけ）。"""
    text = ", ".join(f"{d:%Y-%m-%d}" for d in dates[:MAX_DATES_IN_LOG])
    if len(dates) > MAX_DATES_IN_LOG:
        text += f" ほか {len(dates) - MAX_DATES_IN_LOG} 件"
    return text


def clean_price_data(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    株価データを検査し、誤りの行（終値が 0 以下・すぐに戻る急変）を取り除いて返します。

    見つかったものは、銘柄と日付つきで警告します。戻らない急変は警告だけで、行は残します。

    Args:
        df (pd.DataFrame): Close 列を持つ株価データ（日付順）。
        ticker (str): 警告に出す銘柄名。

    Returns:
        pd.DataFrame: 誤りの行を取り除いた新しい DataFrame（誤りがなければ df のコピー）。
    """
    if df.empty or "Close" not in df.columns:
        return df.copy()
    found = find_price_errors(df)
    if found.non_positive:
        logger.warning(
            f"[{ticker}] 終値が 0 以下の行を {len(found.non_positive)} 行取り除きました: "
            f"{_format_dates(found.non_positive)}"
        )
    if found.reverting_spikes:
        logger.warning(
            f"[{ticker}] 終値が1日で {SPIKE_RATIO:g} 倍以上・{SPIKE_RATIO:g} 分の1以下に動き、"
            f"{MAX_SPIKE_ROWS} 行以内に元に戻った行を、誤りとして {len(found.reverting_spikes)} 行"
            f"取り除きました: {_format_dates(found.reverting_spikes)}"
        )
    if found.jumps:
        logger.warning(
            f"[{ticker}] 終値が1日で {SPIKE_RATIO:g} 倍以上・{SPIKE_RATIO:g} 分の1以下に動き、"
            f"元に戻らない日があります（行は残します。分割の調整漏れなどの可能性があります）: "
            f"{_format_dates(found.jumps)}"
        )
    return df.drop(index=found.dropped)
