# CLAUDE.md

このファイルは、このリポジトリで作業する Claude Code (claude.ai/code) への指針です。

## プロジェクト概要

`vectorbt` ベースの、株式売買戦略のバックテスト用フレームワーク(日本株向け)。
インジケーター、エントリー/エグジットのシグナル、損切り/利確、ポジションサイジングをそれぞれ独立した部品として実装し、組み合わせて `Backtester` に渡してシミュレーションを実行する。
コードベースの docstring とコメントは日本語で書かれている。既存ファイルを編集するときは、この書き方に合わせること。

## 開発環境

- Dev Container 上で開発している。依存パッケージは `.devcontainer/setup.sh` が `uv` で導入する
- Python のコマンドは Dev Container 内で実行する前提

## コマンド

依存パッケージの導入(通常は Dev Container のセットアップ時に自動実行される):
```bash
uv pip install --system -r requirements.txt
```

テスト全体の実行:
```bash
pytest
```

テスト1件の実行:
```bash
pytest tests/test_strategy.py::test_golden_cross_and_take_profit
```

株価を取得せずにキャッシュだけで実行する(Codespaces では先に `python -m tools.cache_sync pull` でキャッシュを取ってくる):
```bash
TRADING_DATA_MODE=offline python -m tools.strategy01
```

メインのバックテストの実行(yfinance で株価を取得・キャッシュし、バックテストを実行して統計を表示、CSV / Google Sheets に出力する):
```bash
python -m tools.strategy01
```

型チェック(Pylance と同じ判定。設定は `pyrightconfig.json` の `standard` モードで、対象は `lib/` / `tools/` / `tests/`):
```bash
npx -y pyright@1.1.408
```
pandas の型は `pandas-stubs`(Pylance の同梱スタブと同じバージョン)で解決する。`pyrightconfig.json` があるため、エディタの Pylance も VS Code 側の型チェックモードの設定ではなく、このファイルを使う。

リント/フォーマットのコマンドは未設定(`pyproject.toml` / `ruff.toml` なし)。Dev Container に VS Code 拡張の `ruff` は入っている。

## 作業ルール

- 返答・説明は日本語で行う
- 大きな変更(複数ファイルにまたがる変更、`Backtester` の挙動が変わる変更)は、実装前に方針を示して確認を取る
- コード変更後は `pytest` を実行し、結果を報告する
- 変更したファイルの Pylance のエラー・警告も、極力解消する
  - 型を直して解消できるものは、型を直す(型注釈の修正、`typing.cast()` など)。ignore で黙らせない
  - vectorbt 由来などで避けられないものだけ、`# pyright: ignore[<ルール名>]` で無視するルールを限定する。行の全ルールを無視する `# type: ignore` は使わない。同じ ignore が何箇所も必要なら、ヘルパーにまとめて1箇所にする
  - `tests/` は vectorbt の結果を直接見る箇所が多いため、ファイルの先頭で `# pyright: reportFunctionMemberAccess=false` を指定してよい(このルールだけをファイル単位で無視する)
  - Pylance はエディタで開いているファイルしか解析しないことがあるため、上記の `pyright` コマンドでも確認する
- indicator / signal / exit / sizer を追加したときは、対応するテストも書く
- 依存パッケージの追加は、先に相談する
- `.env` と `cread/` の中身は読まない・出力しない(認証情報が入っている)

## バックテストの正しさに関するルール(最重要)

- ルックアヘッドバイアス(未来の情報の混入)を入れない。シグナルや SL/TP の `shift(1)` の扱いを変更・削除するときは、理由を説明して確認を取る
- 売買結果に影響する設定(`fees`、`slippage`、`stop_exit_price`、`size_granularity`、`min_size`、約定価格など)を変えるときは、変更前後の結果の差も報告する
- 結果が良くなる変更ほど疑う。過学習や未来情報の混入がないか確認する

## アーキテクチャ

### データフロー

```
DataLoader (lib/data_loader.py) -> Indicators -> Signals -> Exits (SL/TP) -> PositionSizer -> Backtester.run() -> vectorbt.Portfolio
```

`lib/backtester.py` の `Backtester` クラスが全体を取りまとめる。`run(df)` の処理は次のとおり。

1. 各 indicator の `.generate(df)` を呼ぶ。これらは `df` を直接変更し、`short_ma`、`long_ma`、`atr` などの列を追加する。`run()` / `prepare()` は先頭で `df` をコピーするので、呼び出し元の `df` は変わらない(`run()` は列を追加したコピーを `self.df` に残し、`lib.report.chart_table()` / `save_chart_to_sheet()` に渡して使う)。signal / exit / sizer は `df` に列を追加しない
2. 損切り/利確のオブジェクトから `sl` / `tp` の割合を計算し、**1日シフトする**。T日のエントリーに使う損切り/利確の割合は前日終値が基準になるため、「翌日」の約定に合わせてずらしている。`sl` / `tp` は省略でき(`None`)、省略時は全行 NaN になる。「ストップなし」は **NaN** で表し、0 は使わない(vectorbt では幅 0 でもストップは有効で、建値のストップになる)。シフトで空く先頭の行や指標の計算前の期間も NaN のまま残す(0 で埋めない)
3. 全てのエントリーシグナルを **AND** で結合し、全てのエグジットシグナルも **AND** で結合する(`_generate_signals`: `pd.concat(...).all(axis=1)`)。**OR ではない**ので注意。各シグナルは `generate()` の時点で1日シフト済み(下記の `signals/` を参照)
4. `PositionSizer` でポジションサイズを計算する。このとき、シフト済みの `sl_pct` を渡す。あわせて100株単元を調整後の株数に換算する(`_lot_size`: `100 × RawOpen / Open`。`RawOpen` がなければ100)。シミュレーションは調整後価格で行うので、分割・配当で価格が下方修正された過去ほど、実際の100株は調整後の株数では多くなるため
5. `vbt.Portfolio.from_signals(...)` を呼ぶ。日本市場向けの制約がハードコードされている:
   - `size_granularity` / `min_size` は上記の単元で、**エントリーシグナルのある日だけ**かける。単元(調整後の株数)は配当落ち・分割のたびに変わるため、エグジットの日にもかけると全量を売れなくなる
   - `direction="longonly"`
   - `price=df["Open"]`(翌日始値で約定)
   - `stop_exit_price="stopmarket"`(SL/TP の決済にもスリッページをかける。既定値の `stoplimit` では0になる)
   - `log=True`(注文の記録を残す。1単元に届かないなどで捨てられたエントリー注文は取引履歴に残らないため、`portfolio.logs` から数える。売買の結果は変わらない)。`run()` は約定率が `Backtester` の引数 `min_fill_rate`(既定値 `DEFAULT_MIN_FILL_RATE`=0.8、`None` で無効)を下回ると警告する
   - `fees` / `slippage` は `Backtester` の引数で差し替えられる(片道の割合。既定値は `DEFAULT_FEES=0.0`、`DEFAULT_SLIPPAGE=0.001`)。`run_portfolio` でも factory が作る `Backtester` の値を使う
   - `sl_trail` は損切りの `is_trailing` をそのまま渡す(`PreparedSignals.sl_trail`。`sl=None` なら `False`)。トレーリングでは SL ラインが保有中の最高値(`High`)を基準に上がる。損切り幅はエントリー日の `sl_stop` で固定される。vectorbt は各足で判定してから最高値を更新するため、当日の高値で上がったラインが効くのは翌日から
   - vectorbt はエントリーした足で SL/TP を判定しないため、エントリー当日に SL/TP ラインへ届く日だけ、足を2行に分けてから渡す(`PreparedSignals.split_entry_day_bars`)。1行目(元の日時)は始値だけの足で約定し、2行目(`SPLIT_BAR_OFFSET`=12:00)は始値と実際の H/L/C で SL/TP を判定する。判定のラインは、スリッページ込みの約定価格が基準。SL/TP の幅が NaN の側は「なし」として判定しない。SL と TP の両方に届いた日は SL が優先される。`sl_trail=True` のときは、ラインへ届いたかに関係なく、エントリーのある日を全て分ける(分けないとエントリー当日の高値が最高値に入らないため)。シフト済みの値は元の行に残すので、分割は必ず `prepare()` の後に行う(分割後の行に `shift` をかけると、ずれが1日ではなく1行になる)。該当日があると、`Portfolio` のインデックスと決済の `Exit Timestamp` に 12:00 の行が入る

`run()` は `prepare(df)`(上記 1〜4 を行い、1銘柄分の入力を日次のまま `PreparedSignals` で返す。属性は全て `pd.Series`)と、足の分割、`from_signals` の呼び出しに分かれている。日本市場向けの固定条件は `JP_MARKET_PARAMS` にまとめてある(単元の制約は日ごとに変わるため `PreparedSignals` 側、取引コストは差し替えられるように `Backtester.cost_params` 側)。

評価期間は `run(df, start=..., end=...)` / `prepare(df, start=..., end=...)` で指定する(両端を含む。省略時は `df` の全期間)。`prepare()` は 1〜4 を `df` の全期間で計算してから、評価期間の行を切り出す(`PreparedSignals.between`)。評価期間より前の行はインジケーターのウォームアップにだけ使い、口座は評価期間の初日に初期資金で始まるので、年率の指標やエクスポージャーも評価期間だけで計算される(vectorbt の単一銘柄の `Portfolio` は時間で切れず、切っても初期資金が合わないため、指標の側では切らない)。切り出しは必ずシフトの後・足の分割の前に行う(先に切ると、評価期間の初日の指標が NaN になる)。評価期間の初日にインジケーターの列や損切り幅が NaN なら、ウォームアップ不足を警告する。`run()` が残す `self.df` も評価期間の行だけになる。エントリーの期間を絞るだけなら `InDateRangeSignal` も使えるが、年率の指標に取引しない期間が入るため、評価には `start` / `end` を使う

### 複数銘柄・パラメータグリッド(`lib/multi_backtester.py`)

`MultiBacktester(factory, param_grid)` は、パラメータを受け取って新しい `Backtester` を返す関数(factory)を使って、複数銘柄を一括で検証する。

- `run_independent(data, start=None, end=None, benchmark=True)`: 銘柄 × パラメータの全組み合わせを独立した口座で実行し、1行1実行の指標(`lib/metrics.py` の `calc_metrics`)の DataFrame を返す。`benchmark=True` なら、その銘柄の買い持ち(`lib.benchmark.buy_and_hold`。同じ評価期間で、初期資金と取引コストは factory が作る `Backtester` の値。パラメータによらないので銘柄ごとに1回だけ計算する)の `bh_total_return` / `bh_cagr` / `bh_max_drawdown` / `bh_sharpe` と、`excess_cagr`(戦略の CAGR − 買い持ちの CAGR)の列を足す(#8)
- `aggregate(summary, by=None)`: パラメータごと(`by` に列名を渡すと、パラメータ × その列の値ごと。業種・規模別に見るときは `lib.universe.attach_attributes()` で `sector` / `size` を結合してから渡す)に銘柄横断の中央値・平均・プラス銘柄の割合を出す。取引0回の銘柄は中央値・平均・`positive_ratio` から除外し、件数を `zero_trade_tickers` に出す。買い持ちの列があれば、`bh_cagr_median`、`excess_cagr_median` / `excess_cagr_mean`、買い持ちより CAGR が高かった銘柄の割合 `beat_bh_ratio` も出す(取引0回の銘柄の除外は同じ)。保有中のトレードだけの銘柄(`num_trades` が0で `open_trades` が1以上)は取引ありとして扱う(`total_return` に含み損益が入るため)
- `run_portfolio(data, params, start=None, end=None)`: 全銘柄を1つの口座(`cash_sharing=True`、`call_seq="auto"`)で運用する。銘柄ごとの `PreparedSignals` を(足を分割してから)`_combine()` で列=銘柄の DataFrame にまとめ、`from_signals` のキーワード引数(dict)として渡す(`PreparedSignals` には詰め直さない)。`SizeType.Percent` のサイザーは**現金残高**に対する割合になる。同じ日に買いが重なり現金が足りない場合は、**サイズの小さい順**に先着(`call_seq="auto"` が注文の大きさの順に処理するため)。サイズが同じなら `data` の銘柄の並び順で、サイザーの上限(`max_size`)に達した銘柄同士もこれになる
  - `call_seq="auto"` は同じ行の売りを買いより先に処理するため、そのままでは場中の SL/TP の決済の代金が、同じ日の始値の買いに使われる(未来の情報の混入)。そこで、どれかの銘柄にエントリーのある日は、全銘柄の足を2行に分ける(`PreparedSignals.split_bars_for_cash_sharing`)。1行目(元の日時)は**前日終値だけの足**で、エントリーとシグナルの売りが `price`(始値)で約定する。2行目(12:00)は当日の実際の OHLC で、SL/TP を判定する。そのため、SL/TP の決済の代金は(寄り付きのギャップで始値で決済された場合も)**翌日の買いから**使える。シグナルの売りの代金は、同じ日の始値の買いに使える。1行目に前日終値を使うので、保有中の SL/TP は1行目では発動せず、トレーリングの最高値も分けない日と同じように更新される。この分割はエントリー当日の SL/TP 判定(`split_entry_day_bars`)も兼ねるので、`run_portfolio` では `split_entry_day_bars` を使わない
  - そのため、1銘柄だけでも `Backtester.run` とは行数と1行目(始値の時点)の評価額が変わる。売買の結果と日付ごとの終値時点の資産は同じになる(テストでは `tests/helpers.py` の `assert_same_daily_result()` で比べる)。行が増えるぶん、シャープレシオなど行ごとのリターンを使う指標は、日次で計算した値から少しずれる(CAGR は暦日ベースなので変わらない)
- `load_universe(tickers, start, end, source=None)`: `fetch_stock_data` を銘柄ごとに呼んで `{ticker: df}` を返す。取得できなかった銘柄はスキップし、最後に件数をまとめて警告する

### 戦略コンポーネント(`lib/strategies/`)

全てのコンポーネントは同じテンプレートメソッドパターンで作られている。`AbstractX` 基底クラスが `required_columns`(抽象プロパティ)と `_calculate()`(抽象メソッド)を定義し、公開メソッドの `generate()` が必要な列の存在を検証してから `_calculate()` に処理を渡す。列の検証は `lib/validation.py` の `validate_required_columns()` に共通化してあり(`AbstractPriceSource.fetch()` も同じ)、不足している列をクラス名つきでまとめて表示する。新しい基底クラスでも、ループを書かずにこれを使う。
新しい indicator / signal / exit / sizer を追加するときは、該当サブフォルダの抽象クラスを継承し、そのサブフォルダの `__init__.py` からエクスポートすること。

- `indicators/`(`AbstractIndicator`): 株価の DataFrame に列を追加する。例: `MAIndicator` は `pandas_ta` で `short_ma` / `long_ma` を、`ATRIndicator` は `atr` を追加する。出力する列名は引数で変えられ(`short_col` / `long_col`、`col`。既定値は上記の名前)、追加する列は抽象プロパティ `output_columns` で宣言する。`generate()` は既にある列への出力を `ValueError` にし(黙って上書きしない)、`Backtester.__init__` も indicator 同士の出力列の重複を `ValueError` にする。同じ種類の indicator を2つ使うときは列名を変え、それを読む signal / exit にも同じ列名を渡す(`short_col` / `long_col`、`atr_col`)。`ExternalIndicator(source, indicator, name)` は、別の銘柄(TOPIX 連動 ETF など)の株価データ `source` で既存の indicator を計算し、その出力列を売買する銘柄の行に日付で合わせて追加する(地合いフィルタ用)。`source` の全期間で計算してから合わせ、`source` に行がない日は前の日の値で埋める(過去方向のみ)。`source` の最終日より後の日は NaN にして警告する(古いキャッシュの値を使い続けないため)。T日の行には `source` のT日の値が入るので、読むシグナルは通常どおり1日シフトする
- `signals/`(`AbstractSignal`): エントリー/エグジットの条件を、bool の `pd.Series` で返す。例: `GoldenCrossSignal`、`DeadCrossSignal`、`InDateRangeSignal`、`ATRConditionSignal`、`ColumnAboveSignal`(`left_col > right_col` の間ずっと True。クロスの日だけの `GoldenCrossSignal` と違い、フィルタに使う。NaN の日は False)。サブクラスの `_calculate()` はシフトせずに「T日の終値で分かる情報でT日の値」を返し、`AbstractSignal.generate()` が一括で1日シフトする(T日の終値で出たシグナルは、T+1日の始値で約定する)。日付だけで決まるものなど、T日の寄り付き前に分かるシグナルは、クラス属性 `shift_required = False` を宣言してシフトを止める(例: `InDateRangeSignal`)。既定値は `True` なので、宣言を忘れても1日遅れるだけで未来参照にはならない。`tests/test_signal_shift.py` の `_all_signals()` に新しいシグナルを足すと、未来の価格を書き換えて過去のシグナルが変わらないかを確かめる汎用テストの対象になる
- `exits/`(`AbstractStopLoss` / `AbstractTakeProfit`): `Close` に対する比率の Series を返し、vectorbt の `sl_stop` / `tp_stop` として使われる。例: `AtrStopLoss` / `AtrTakeProfit` は `atr * multiplier / Close`
- `position_sizing/`(`AbstractPositionSizer`): ポジションサイズを `pd.Series` で返す。`size_type`(`vectorbt.portfolio.enums.SizeType`)を宣言し、`vbt.Portfolio.from_signals` にそのまま渡される。`RiskPercentageSizer` は `risk_per_trade / sl_pct` を計算する(資産に対するリスク割合でのサイジング。kwargs で `sl_pct` を渡す必要がある)。損切り幅が NaN / 0 以下の日はサイズを NaN にして注文を出さない。損切り幅を使うサイザーはクラス属性 `requires_sl_pct = True` を宣言する。サイズの上限は `AbstractPositionSizer` の `max_size`(単位は `size_type` に従う。既定値は `None` = 上限なし)で、`generate()` が `clip` をかける。`RiskPercentageSizer` の既定値は 1.0(現金の100%。vectorbt は超えた分を黙って現金全額に丸めるため)。`Backtester.prepare(df, name)` は、エントリーの日のうちサイズが上限に達した件数を警告する(その分、実際のリスクは `risk_per_trade` より小さい)。`Backtester` は `sl=None` と組み合わせると `ValueError` にする

### データ層

- `lib/data_sources/`(`AbstractPriceSource`): 株価データの取得元。`name`(キャッシュのフォルダ名)と `_fetch()` を実装し、公開メソッドの `fetch()` が必須列(`Open/High/Low/Close/Volume` は分割・配当の調整後、`RawOpen` は当時の実際の始値)を検証する。取得元ごとに必要な価格の補正も、ここで行う(補正だけを別の部品にはしない)。現在は `YFinanceSource` のみで、`auto_adjust=True` と同じ調整後価格を作り、`RawOpen` は `Stock Splits` から分割比率を掛け戻して作る(yfinance の未調整の価格も分割調整済みのため)。将来 J-Quants API などに差し替えられるようにしている
- `lib/data_loader.py`: `fetch_stock_data(ticker, start, end, source=None, mode=None, max_age_days=None)` が取得元(省略時は `YFinanceSource`)から取得し、`data/<取得元の名前>/<ticker>.parquet` にキャッシュする(必須列のない古いキャッシュは全期間を取り直す)。毎回全期間を取り直さず、キャッシュ済みの最終日以降だけを取得して、結合・重複排除する差分更新を行う。取得モード(#66)は引数 `mode`、環境変数 `TRADING_DATA_MODE`、既定値 `auto` の順に決まる(`refresh=True` は `mode="refresh"` と同じ)
  - `auto`: 最後に取得元へ問い合わせてから `max_age_days`(環境変数 `TRADING_CACHE_MAX_AGE_DAYS`、既定値7日)以内なら問い合わせない。取得に失敗したら、キャッシュがあれば警告してそれを使う
  - `offline`: 取得元に一切問い合わせない。キャッシュがなければ `FileNotFoundError`(`load_universe()` ではスキップ)
  - `refresh`: キャッシュを使わず全期間を取り直す
  - 最終取得時刻は `<ticker>.meta.json` に記録する(ないキャッシュは期限切れ扱い)。結果がキャッシュの時点で変わるため、銘柄ごとのデータの最終日をログに出す(`load_universe()` は最古・最新をまとめて出す)
  - 返す前に、期間で絞ったデータを `lib/data_quality.py` の `clean_price_data()` で検査する(`check_quality=False` で止められる。キャッシュには取得元のデータをそのまま保存する)
- `lib/data_quality.py`: 株価データの検査(#70)。終値が 0 以下の行と、終値が1日で `SPIKE_RATIO`(2倍)以上・分の1以下に動いて `MAX_SPIKE_ROWS`(5行)以内に元の水準(`REVERT_TOLERANCE`=±25%)へ戻った行を、誤りとして**取り除き**、銘柄・日付つきで警告する(1306.T の 2026-03-30・03-31 が本来の 1/10 になっているなど、yfinance のデータの誤りのため)。前の値で埋めないのは、実際にはなかった値動きのない日を作らないため。戻らない急変は本当の値動きの可能性があるので、警告だけで残す。戻ったかの判定は後の行を見るが、価格の誤りを直すためでシグナルには使わない(最後の行の急変は警告だけ)
- `lib/data_saver.py`: `gspread` で DataFrame を Google Sheets に書き込む。環境変数 `SPREADSHEET_ID` / `CREDENTIALS_FILE`(`.env` から読み込み)で遅延初期化する。`lib/report.py` の `save_chart_to_sheet()` / `save_orders_to_sheet()` / `save_trades_to_sheet()` はこれに依存しており、スプレッドシートが未設定だと例外になる(`data_saver` は関数の中で import するので、表示・CSV だけなら gspread も `.env` も要らない)
- `lib/report.py`: バックテスト結果(`vbt.Portfolio`)の出力。`print_summary()`(サマリと最新の注文の表示)、`orders_table()` / `trades_table()`(`records_readable`。`pyright: ignore` はここにまとめてある)、`save_orders_csv()` / `save_trades_csv()`、`chart_table()`(株価・指標、シグナル、日付ごとに集約した約定を結合した表。12:00 の行の決済は日付にそろえる。複数銘柄の `Portfolio` では `column` で銘柄を絞る)、Sheets への保存（任意の表は `save_table_to_sheet()`。NaN と ±inf は空欄にする。inf は Sheets API に書けないため。出力先が設定されているかは `sheet_is_configured()`）、`show()`。`Backtester` は出力の処理を持たないので、`run()` / `run_portfolio()` の結果をここに渡す。出力の種類を足すときも、ここに関数を足す
- `lib/metrics.py`: 成績の指標の計算。`year_freq(index)` は、行数 ÷ 暦年から1年あたりの本数を求める(vectorbt の既定値は `freq="D"` で365本/年で、営業日だけの日足では CAGR が小さく、シャープレシオが大きく出るため)。`calc_metrics()`(一括検証の指標)と `portfolio_stats()`(`stats()` の表示。`lib.report.print_summary()` などで使う)は、どちらもこの基準で年率換算する。年率の指標を新しく表示・計算するときも、`portfolio.stats()` などを直接呼ばずにこれらを使う。`entry_order_stats()` は、列(銘柄)ごとにエントリー注文の数・約定数・捨てられた数と理由(`OrderStatusInfo`)を返す(保有中のシグナルは注文にならないので数えない)。`calc_metrics()` の `entry_orders` / `entry_dropped` / `entry_fill_rate` もこれを使う。サイザーを追加したときは、この値で注文が通っているか確認する。`calc_metrics()` のトレード単位の指標(`num_trades`、`win_rate`、`profit_factor`、`expectancy`、`avg_trade_return`)は**決済済みのトレードだけ**で計算し(`trades.closed`。`portfolio_stats()` の Win Rate なども同じ)、期間の最後に保有中のトレードは `open_trades` に数える。含み損益で勝ち負けを数えると、データの終了日だけで値が変わるため。`total_return` などの口座単位の指標は含み損益を含む
- `lib/benchmark.py`: 買い持ちのベンチマーク(#8)。`buy_and_hold(df, start, end, init_cash, fees, slippage)` は評価期間の初日の**始値**(スリッページ込み)で現金の全額を買い、最終日の終値で評価する(最後に売らない)。`equal_weight_buy_and_hold(data, ...)` は複数銘柄を初期資金の等分で買って持ち続ける(リバランスなし、資金共有の1口座。評価期間の途中で上場した銘柄の分は上場日まで現金)。どちらも調整後価格(配当込み)で、100株単元はかけず端株も買う(単元をかけると、使えない現金の比率でリターンが変わるため)。指標は `lib.metrics.account_metrics()`(`calc_metrics()` の口座単位の指標 `total_return` / `cagr` / `max_drawdown` / `sharpe` / `sortino` を切り出したもの)で計算するので、戦略と同じ基準で比べられる。vectorbt の `stats()` の `Benchmark Return` は終値から終値でコストもないので、比較には使わない。市場全体のベンチマークは `BENCHMARK_TICKER`(1306.T、TOPIX 連動 ETF)。戦略の `Portfolio` には足を分けた 12:00 の行が入るため、シャープレシオなど行ごとのリターンを使う指標は、日次の買い持ちと少しずれる(#40。CAGR と最大ドローダウンはずれない)。`lib.report.benchmark_table()` / `print_benchmark_comparison()` が、戦略と各ベンチマークの指標を列に並べた表を作る・表示する
- `lib/universe.py`: 検証する銘柄ユニバースの管理(#13)。ユニバースは `universes/<名前>.csv`(UTF-8。`.gitignore` の `*.csv` の例外として Git で管理する)に置き、必須列は `ticker` / `name` / `sector`(東証33業種)/ `size`(TOPIX の規模区分)。`load_universe_file()` が列・`ticker` の空と重複を検証して読み込む(行の順序はそのまま。資金共有モードでサイズが同じ買いが重なったときの順序になる)。`has_survivorship_bias()` は、上場廃止日の列(`delisted_on`)がない・全て空なら True(今のユニバースは全てこれ)。任意の列 `data_from` は、銘柄ごとにデータを使い始める日(上場前のデータが壊れている銘柄のため。8919.T は 2017-12-12)で、`data_from_map()` で辞書にして `load_universe(..., data_from=...)` に渡すと、`start` とその日の遅い方から取得する。`jpx_stratified_100.csv` では `tools/build_universe.py` の `DATA_FROM_OVERRIDES` から作る。`lib.report.print_universe_note()` がユニバースの名前・銘柄数と、生存者バイアスの注意書きを表示する。`universes/jpx_stratified_100.csv` は `tools/build_universe.py` が JPX の東証上場銘柄一覧(`universes/source/`、UTF-8 の CSV に変換したもの。`python -m tools.build_universe --download` でダウンロード・変換する。JPX の利用規約で再配信が禁じられているため gitignore しており、`tools/cache_sync.py` で共有する)から、規模区分ごとの件数(`SIZE_QUOTAS`)を33業種で層化して固定のシードで選んだもの。`tests/test_universe.py` が、作り直した結果と commit 済みのファイルの一致を確かめる(一覧がない CI ではスキップ)ので、設定や一覧を変えたら作り直して commit する。`universes/large5.csv` は以前の検証銘柄(大型株5銘柄)
- `lib/__init__.py`: `setup_logging()` が `config/logging_config.json` を読み込む(`lib` と `tools` の名前空間のロガーを定義)。新しいエントリーポイントのスクリプトでは、ログ出力の前に1回呼ぶこと

### エントリーポイント・テスト

- `tools/strategy01_multi.py`: strategy01 を、ユニバース(`UNIVERSE_NAME`。既定値は `jpx_stratified_100`)の銘柄 × 小さなパラメータグリッドで回すサンプル(CSV のみ出力)。パラメータごとの集計に加えて、規模区分・業種ごとの集計も出す。CSV には `universe` 列で、どのユニバースの結果かを残す。株価はユニバースの `data_from` の日から読み込む。グリッドの `market_filter` で地合いフィルタ(1306.T の50日線 > 200日線の日だけ買う。`ExternalIndicator` + `ColumnAboveSignal`。決済には使わない)の有無を比べ、資金共有モードもフィルタありを比較表に並べる。1306.T は 2015-01-05 に戻らない -90% の足があるため、`MARKET_DATA_FROM`(2015-01-06)より後のデータだけを使う。銘柄ごとの買い持ちとの比較(`excess_cagr_median`、`beat_bh_ratio`)を表示し、資金共有モードはユニバースの等金額の買い持ち(銘柄の選び方・生存者バイアスを除いた戦略だけの差を見る)と 1306.T の買い持ちと並べて、`multi_benchmark.csv`(シートは `Multi_Benchmark`)に出す。`SAVE_TO_SHEET` が True なら、同じ表と資金共有モードの取引履歴を `Multi_` で始まるシートにも上書きで出力する(`strategy01` の `Chart` / `Orders` / `Trades` とぶつけないため。`SPREADSHEET_ID` が未設定なら警告して飛ばす)
- `tools/build_universe.py`: `universes/jpx_stratified_100.csv` を作るスクリプト(上記の `lib/universe.py` を参照)
- `tools/cache_sync.py`: 株価のキャッシュ(`data/yfinance/`)と JPX の一覧(`universes/source/`)を1つの zip にまとめ、private のデータ用リポジトリ(`taki036731/trading-data`、環境変数 `TRADING_DATA_REPO` で変更可)のリリース(タグ `cache-latest`)と `gh` で共有する(#66)。`push` は手元の PC で、`pull` は Codespaces などで手動で実行する。このリポジトリは public なので、Yahoo・JPX のデータはここ(コミット・LFS・リリース・Actions artifact)に置かない。Codespaces から読む権限は `.devcontainer/devcontainer.json` の `customizations.codespaces.repositories` で宣言している
- `tools/strategy01.py`: 基準となるサンプル。EMA 5/20 のゴールデン/デッドクロス、ATR のボラティリティフィルター、ATR ベースの損切り/利確、リスク割合サイジングを組み合わせ、`7203.T`(トヨタ)に対して実行し、7203.T と 1306.T の買い持ちと並べて表示する。データは2010年から取得し、評価期間は `EVAL_START`(2016-06-25)以降。新しい戦略スクリプトを `tools/` に作るときのテンプレートとして使う
- `tests/helpers.py` の `create_base_df()`: 戦略のテスト用に、`short_ma` / `long_ma` / `atr` の列があらかじめ入った合成 OHLC の DataFrame を作る(indicator のステップを通さない)

## その他

- `.github/workflows/merge-code.yml` は `main` への push ごとに、全ての `*.py` / `*.sql` を結合して `latest_code.md` を作り、`rclone` で Google Drive にアップロードする(Gemini に読ませる用)。CI ではない
- `.github/workflows/test.yml` は `main` への push、全ての PR、手動実行のときに、`pytest` と `pyright` を実行する(CI)。`pyright` は `pytest` が失敗しても実行する
- 認証情報(`SPREADSHEET_ID`、`CREDENTIALS_FILE`)は `.env`(gitignore 済み)にある。`cread/` には Google サービスアカウントの JSON 認証ファイルがある(gitignore 済み)
