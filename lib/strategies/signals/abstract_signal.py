from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class AbstractSignal(ABC):
    """
    売買シグナル判定の抽象基底クラス。
    特定の条件（ゴールデンクロス、期間指定など）に基づいてTrue/FalseのSeriesを生成します。

    サブクラスの `_calculate()` は「T日の終値の時点で分かる情報」でT日の値を返します。
    約定は翌日始値なので、`generate()` がこれを1日シフトし、T日の値をT+1日のシグナルにします。
    シフトを各サブクラスに任せると、書き忘れたときに当日終値で判断して当日始値で買う
    未来参照になるため、シフトはこのクラスで一括して行います。

    日付だけで決まる条件など、T日の寄り付き前にT日の値が分かるシグナルは、
    クラス属性 `shift_required = False` を宣言してシフトを止めます。
    既定値は True（シフトする）です。宣言を忘れても、シグナルが1日遅れるだけで未来参照にはなりません。
    """

    shift_required: ClassVar[bool] = True
    """`_calculate()` の結果を1日シフトするか。T日の寄り付き前に分かる値なら False にします。"""

    def __init__(self, **kwargs):
        """
        シグナルのパラメータを初期化します。

        Args:
            **kwargs: シグナル判定に使用する任意のパラメータ。
        """
        self.params = kwargs

    @abstractmethod
    def _calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        具体的なシグナル判定ロジックを実装します。
        結果はシフトせず、T日の終値の時点で分かる情報で、T日の値を返します（シフトは `generate()` が行います）。

        Args:
            df (pd.DataFrame): 判定に必要なカラム（指標など）を含むデータフレーム。

        Returns:
            pd.Series: 各行がTrue（条件合致）またはFalse（条件不一致）のブール値Series。
        """

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        シグナル判定に必須なカラム名のリストを定義します。

        Returns:
            list[str]: 必須カラム名のリスト。
        """

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        バリデーションを実行した後、シグナルを生成します。
        `shift_required` が True なら、結果を1日シフトします（先頭は False）。

        Args:
            df (pd.DataFrame): 株価および指標データ。

        Returns:
            pd.Series: 生成されたシグナルのブール値Series（その日の始値で約定してよいか）。

        Raises:
            ValueError: 必須カラムが不足している場合。
        """
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"エラー: 必須カラム'{col}'が存在しません。")
        signal = self._calculate(df).fillna(False).astype(bool)
        if self.shift_required:
            signal = signal.shift(1, fill_value=False)
        return signal
