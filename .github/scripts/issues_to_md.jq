# gh issue list --json の出力を、1つの Markdown にまとめる
# 引数: $repo (owner/name), $now (更新日時の文字列)

# ISO8601 (UTC) の日時を JST の "YYYY-MM-DD HH:MM" に変換する
def jst: fromdateiso8601 + 9 * 3600 | strftime("%Y-%m-%d %H:%M");
def label_text: if (.labels | length) > 0 then [.labels[].name] | join(", ") else "なし" end;
def body_text: if (. // "") == "" then "(本文なし)" else . end;

# OPEN を先、そのあと CLOSED。それぞれ番号の降順
sort_by([(.state != "OPEN"), -.number]) as $issues
| [
    "# GitHub Issues まとめ (\($repo))",
    "更新日時: \($now) (JST)",
    "",
    "## 目次",
    ($issues[] | "- #\(.number) [\(.state)] \(.title)" + (if (.labels | length) > 0 then " `" + ([.labels[].name] | join("`, `")) + "`" else "" end)),
    "",
    ($issues[] |
      "---",
      "## #\(.number) \(.title)",
      "- 状態: \(.state) / ラベル: \(label_text) / 作成者: @\(.author.login)",
      "- 作成: \(.createdAt | jst) / 更新: \(.updatedAt | jst)" + (if .closedAt then " / クローズ: \(.closedAt | jst)" else "" end),
      "- URL: \(.url)",
      "",
      (.body | body_text),
      "",
      (if (.comments | length) > 0 then
        "### コメント (\(.comments | length)件)",
        "",
        (.comments[] | "**@\(.author.login)** (\(.createdAt | jst)):", "", (.body | body_text), "")
      else empty end)
    )
  ]
| join("\n")
