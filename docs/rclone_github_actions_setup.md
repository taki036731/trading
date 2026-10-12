# rcloneを使ってGitHub ActionsからGoogle Driveへファイルをアップロードする手順

Google Driveのサービスアカウント（容量0GB仕様）による「403 Storage Quota」エラーを回避し、**OAuth認証（個人アカウントの容量）**を利用してGitHub Actionsから安全・確実に自動アップロードを行うためのセットアップ手順です。

---

## 前提条件
- 手元のPC（ブラウザが起動できる環境）に `rclone` がインストールされていること
- アップロード先のGoogleアカウント（@gmail.com）にアクセスできること

---

## ステップ1：手元のPCで認証トークンを発行する

GitHub Actions（ブラウザが開けない環境）であなたのGoogleアカウントの容量を使うため、まずは手元のPCで1度だけログイン認証を行い、設定情報を抽出します。

1. ターミナル（またはコマンドプロンプト）を開き、以下のインタラクティブ設定コマンドを実行します。
   ```bash
   rclone config
   ```

2. 画面の指示に従って以下のように入力します。
   - `n) New remote` を選択（`n` を入力してエンター）
   - **name:** リモート名を入力（例: `gdrive`）
   - **Storage Type:** `drive`（または Google Drive に対応する番号）を選択
   - **client_id / client_secret:** 何も入力せずそのままエンター（rcloneのデフォルトを使用）
   - **scope:** `1`（Full access all file metadata and file contents）を選択
   - **service_account_file:** **何も入力せずそのままエンター**（空欄にすることが最重要です）
   - **Edit advanced config?:** `n` (No) を選択
   - **Use web browser to automatically authenticate rclone with remote?:** `y` (Yes) を選択

3. 自動的にWebブラウザが起動するので、対象のGoogleアカウントでログインし、rcloneへのアクセス権限を**許可（Allow）**します。

4. ブラウザに「Success!」と表示されたらターミナルに戻り、`y) Yes this is OK` を選択して設定を保存します。

---

## ステップ2：認証トークン（rclone.conf）の抽出

1. 以下のコマンドを実行して、作成された設定ファイルの保存場所を確認します。
   ```bash
   rclone config file
   ```
   *表示されたパス（例: `~/.config/rclone/rclone.conf`）のファイルをテキストエディタで開きます。*

2. ファイル内にある、先ほど作成したリモート名（例: `[gdrive]`）のブロックを**base64に変換した上ですべてコピー**して手元に控えます。
   ```ini
   [gdrive]
   type = drive
   scope = drive
   token = {"access_token":"xxxx","token_type":"Bearer","refresh_token":"xxxx","expiry":"xxxx"}
   ```
   *(※このトークンに含まれる `refresh_token` により、GitHub Actions実行時に自動で期限が更新されるため、再認証は不要になります)*

   ```bash
   base64 ~/.config/rclone/rclone.conf | clip.exe
   ```
   *WSL2上であれば、 `clip.exe` に渡すことで、クリップボードにコピーされる*

---

## ステップ3：GitHub Secretsへの登録

GitHub Actions側に認証情報を安全に渡すため、リポジトリの暗号化環境変数（Secrets）に登録します。

1. 対象のGitHubリポジトリのWebページを開きます。
2. **Settings** ＞ **Secrets and variables** ＞ **Actions** の順に進みます。
3. **New repository secret** ボタンをクリックし、以下を設定します。
   - **Name:** `RCLONE_CONF`
   - **Value:** [ステップ2] でBASE64に変換&コピーした `[gdrive]` から始まる設定ブロックの内容をそのまま丸ごと貼り付けます。

---

## ステップ4：アップロードコマンドの指定方法

GitHub Actionsのワークフロー内（ステップ3で復元した設定ファイルを使用する環境）で、以下の形式でコマンドを実行します。

### 基本コマンド
```bash
rclone copy /パス/to/ローカルファイル gdrive:リモートのフォルダパス
```

### パス指定のルールとテクニック
- **マイドライブ直下の指定:** `gdrive:` の直後が自動的にマイドライブのルート階層を指します。
- **階層（サブフォルダ）の指定:** スラッシュ（`/`）で区切って指定します。
  - 例：`gdrive:_WORK/files`
  - 指定したフォルダ階層がGoogle Drive側に存在しない場合は、rcloneが実行時に**自動的にフォルダを新規作成**します。
- **フォルダIDによるピンポイント指定:** 同名フォルダの誤認を防ぎたい場合は、リモート名にIDを直接紐付けることも可能です。
  - 例：`gdrive,id=1A2B3C4D5E6F...:` （末尾のコロンを忘れないよう注意してください）