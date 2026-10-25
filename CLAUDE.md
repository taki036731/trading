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

メインのバックテストの実行(yfinance で株価を取得・キャッシュし、バックテストを実行して統計を表示、CSV / Google Sheets に出力する):
```bash
python -m tools.strategy01
```

リント/フォーマットのコマンドは未設定(`pyproject.toml` / `ruff.toml` なし)。Dev Container に VS Code 拡張の `ruff` は入っている。

## 作業ルール

- 返答・説明は日本語で行う
- 大きな変更(複数ファイルにまたがる変更、`Backtester` の挙動が変わる変更)は、実装前に方針を示して確認を取る
- コード変更後は `pytest` を実行し、結果を報告する
- indicator / signal / exit / sizer を追加したときは、対応するテストも書く
- 依存パッケージの追加は、先に相談する
- `.env` と `cread/` の中身は読まない・出力しない(認証情報が入っている)

## バックテストの正しさに関するルール(最重要)

- ルックアヘッドバイアス(未来の情報の混入)を入れない。シグナルや SL/TP の `shift(1)` の扱いを変更・削除するときは、理由を説明して確認を取る
- 売買結果に影響する設定(`fees`、`size_granularity`、`min_size`、約定価格など)を変えるときは、変更前後の結果の差も報告する
- 結果が良くなる変更ほど疑う。過学習や未来情報の混入がないか確認する

## アーキテクチャ

### データフロー

```
DataLoader (lib/data_loader.py) -> Indicators -> Signals -> Exits (SL/TP) -> PositionSizer -> Backtester.run() -> vectorbt.Portfolio
```

`lib/backtester.py` の `Backtester` クラスが全体を取りまとめる。`run(df)` の処理は次のとおり。

1. 各 indicator の `.generate(df)` を呼ぶ。これらは `df` を直接変更し、`short_ma`、`long_ma`、`atr` などの列を追加する
2. 損切り/利確のオブジェクトから `sl` / `tp` の割合を計算し、**1日シフトする**。T日のエントリーに使う損切り/利確の割合は前日終値が基準になるため、「翌日」の約定に合わせてずらしている
3. 全てのエントリーシグナルを **AND** で結合し、全てのエグジットシグナルも **AND** で結合する(`_generate_signals`: `pd.concat(...).all(axis=1)`)。**OR ではない**ので注意
4. `PositionSizer` でポジションサイズを計算する。このとき、シフト済みの `sl_pct` を渡す
5. `vbt.Portfolio.from_signals(...)` を呼ぶ。日本市場向けの制約がハードコードされている:
   - `size_granularity=100`、`min_size=100`(100株単元)
   - `direction="longonly"`
   - `price=df["Open"]`(翌日始値で約定)
   - `fees=0.0`(**意図的な暫定値**。将来、手数料・スリッページを導入する予定)

`run()` は `prepare(df)`(上記 1〜4 を行い、`PreparedSignals` を返す)と `from_signals` の呼び出しに分かれている。日本市場向けの固定条件は `JP_MARKET_PARAMS` にまとめてある。

### 複数銘柄・パラメータグリッド(`lib/multi_backtester.py`)

`MultiBacktester(factory, param_grid)` は、パラメータを受け取って新しい `Backtester` を返す関数(factory)を使って、複数銘柄を一括で検証する。

- `run_independent(data)`: 銘柄 × パラメータの全組み合わせを独立した口座で実行し、1行1実行の指標の DataFrame を返す。CAGR・シャープレシオは、営業日の本数から求めた1年あたりの本数で年率換算する
- `aggregate(summary)`: パラメータごとに銘柄横断の中央値・平均・プラス銘柄の割合を出す
- `run_portfolio(data, params)`: 全銘柄を1つの口座(`cash_sharing=True`、`call_seq="auto"`)で運用する。`SizeType.Percent` のサイザーは**現金残高**に対する割合になる。同じ日に買いが重なり現金が足りない場合は、`data` の銘柄の並び順で先着
- `load_universe(tickers, start, end, source=None)`: `fetch_stock_data` を銘柄ごとに呼んで `{ticker: df}` を返す

### 戦略コンポーネント(`lib/strategies/`)

全てのコンポーネントは同じテンプレートメソッドパターンで作られている。`AbstractX` 基底クラスが `required_columns`(抽象プロパティ)と `_calculate()`(抽象メソッド)を定義し、公開メソッドの `generate()` が必要な列の存在を検証してから `_calculate()` に処理を渡す。
新しい indicator / signal / exit / sizer を追加するときは、該当サブフォルダの抽象クラスを継承し、そのサブフォルダの `__init__.py` からエクスポートすること。

- `indicators/`(`AbstractIndicator`): 株価の DataFrame に列を追加する。例: `MAIndicator` は `pandas_ta` で `short_ma` / `long_ma` を、`ATRIndicator` は `atr` を追加する
- `signals/`(`AbstractSignal`): エントリー/エグジットの条件を、bool の `pd.Series` で返す。例: `GoldenCrossSignal`、`DeadCrossSignal`、`InDateRangeSignal`、`ATRConditionSignal`。クロス系のシグナルは結果を1日シフトする(N日の終値で出たシグナルは、N+1日のエントリーになる)
- `exits/`(`AbstractStopLoss` / `AbstractTakeProfit`): `Close` に対する比率の Series を返し、vectorbt の `sl_stop` / `tp_stop` として使われる。例: `AtrStopLoss` / `AtrTakeProfit` は `atr * multiplier / Close`
- `position_sizing/`(`AbstractPositionSizer`): ポジションサイズを `pd.Series` で返す。`size_type`(`vectorbt.portfolio.enums.SizeType`)を宣言し、`vbt.Portfolio.from_signals` にそのまま渡される。`RiskPercentageSizer` は `risk_per_trade / sl_pct` を計算する(資産に対するリスク割合でのサイジング。kwargs で `sl_pct` を渡す必要がある)

### データ層

- `lib/data_sources/`(`AbstractPriceSource`): 株価データの取得元。`name`(キャッシュのフォルダ名)と `_fetch()` を実装し、公開メソッドの `fetch()` が必須列(`Open/High/Low/Close/Volume`、価格は分割・配当の調整後)を検証する。取得元ごとに必要な価格の補正も、ここで行う。現在は `YFinanceSource`(`auto_adjust=True`)のみ。将来 J-Quants API などに差し替えられるようにしている
- `lib/data_loader.py`: `fetch_stock_data(ticker, start, end, source=None)` が取得元(省略時は `YFinanceSource`)から取得し、`data/<取得元の名前>/<ticker>.parquet` にキャッシュする。毎回全期間を取り直さず、キャッシュ済みの最終日以降だけを取得して、結合・重複排除する差分更新を行う
- `lib/data_saver.py`: `gspread` で DataFrame を Google Sheets に書き込む。環境変数 `SPREADSHEET_ID` / `CREDENTIALS_FILE`(`.env` から読み込み)で遅延初期化する。`Backtester.save_chart_to_sheet()` / `save_orders_to_sheet()` / `save_trades_to_sheet()` はこれに依存しており、スプレッドシートが未設定だと例外になる
- `lib/__init__.py`: `setup_logging()` が `config/logging_config.json` を読み込む(`lib` と `tools` の名前空間のロガーを定義)。新しいエントリーポイントのスクリプトでは、ログ出力の前に1回呼ぶこと

### エントリーポイント・テスト

- `tools/strategy01_multi.py`: strategy01 を複数銘柄 × 小さなパラメータグリッドで回すサンプル(CSV のみ出力)
- `tools/strategy01.py`: 基準となるサンプル。EMA 5/20 のゴールデン/デッドクロス、期間フィルター、ATR のボラティリティフィルター、ATR ベースの損切り/利確、リスク割合サイジングを組み合わせ、`7203.T`(トヨタ)に対して実行する。新しい戦略スクリプトを `tools/` に作るときのテンプレートとして使う
- `tools/fetch_prices.py` / `sample/from_pine.py`: 古いスクリプト。`lib/data_loader.py` に存在しない関数(例: `dl.add_sma`)を参照している可能性があるため、最新の実装として扱わない
- `tests/helpers.py` の `create_base_df()`: 戦略のテスト用に、`short_ma` / `long_ma` / `atr` の列があらかじめ入った合成 OHLC の DataFrame を作る(indicator のステップを通さない)

## その他

- `.github/workflows/merge-code.yml` は `main` への push ごとに、全ての `*.py` / `*.sql` を結合して `latest_code.md` を作り、`rclone` で Google Drive にアップロードする(Gemini に読ませる用)。CI ではなく、テストやリントの GitHub Action はない
- 認証情報(`SPREADSHEET_ID`、`CREDENTIALS_FILE`)は `.env`(gitignore 済み)にある。`cread/` には Google サービスアカウントの JSON 認証ファイルがある(gitignore 済み)
