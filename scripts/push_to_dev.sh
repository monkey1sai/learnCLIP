#!/usr/bin/env bash
# push_to_dev.sh
# 情境 A：當前已在 dev 分支，將本地變更加入、提交、rebase origin/dev 並推送到遠端。
# 使用方式：
#   ./scripts/push_to_dev.sh "你的 commit 訊息"
# 若未提供 commit 訊息，會提示你輸入。

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"
cd "$REPO_ROOT"

# 檢查是否是 dev 分支
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "<unknown>")
if [[ "$CURRENT_BRANCH" != "dev" ]]; then
  echo "錯誤：目前分支為 '$CURRENT_BRANCH'，此腳本必須在 'dev' 分支上執行。"
  echo "請切換到 dev 後再執行，或使用其他流程將變更移到 dev。"
  exit 1
fi

# 顯示未追蹤/未加入變更摘要
STATUS_PORCELAIN=$(git status --porcelain)
if [[ -z "$STATUS_PORCELAIN" ]]; then
  echo "沒有發現未提交的變更，無需推送。"
  exit 0
fi

echo "以下檔案將被加入並提交："
git status --short
git add -A
# 解析參數：接受 -m "commit message"，其餘位置參數視為要 add 的檔案路徑
COMMIT_MSG=""
FILES=()
usage(){
  echo "Usage: $0 -m \"commit message\" [paths...]" >&2
  echo "If no paths provided, the script will git add -A (all changes)." >&2
}

while getopts ":m:h" opt; do
  case $opt in
    m) COMMIT_MSG="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND -1))
FILES=("$@")

if [[ -z "$COMMIT_MSG" ]]; then
  read -r -p "請輸入 commit 訊息: " COMMIT_MSG
  if [[ -z "$COMMIT_MSG" ]]; then
    echo "未提供 commit 訊息，取消。"
    exit 1
  fi
fi

# 顯示將要提交的檔案供確認
if [[ ${#FILES[@]} -gt 0 ]]; then
  echo "將加入以下指定檔案："
  for f in "${FILES[@]}"; do
    echo "  $f"
  done
else
  echo "沒有指定檔案，將 add 所有變更（git add -A）。"
  git status --short
fi

echo
read -r -p "確認要 add & commit 並 push 到 origin/dev？(y/N): " CONF
if [[ "$CONF" != "y" && "$CONF" != "Y" ]]; then
  echo "已取消。"
  exit 0
fi

# add & commit
if [[ ${#FILES[@]} -gt 0 ]]; then
  git add -- "${FILES[@]}"
else
  git add -A
fi
# 如果沒有變更，git commit 會失敗；但我們已經檢查過有變更。
if git commit -m "$COMMIT_MSG"; then
  echo "commit 完成：$(git rev-parse --short HEAD)"
else
  echo "commit 失敗，請檢查 git 狀態。"
  exit 1
fi

# 同步遠端並 rebase
echo "fetch origin 並對 dev 執行 rebase..."
git fetch origin
if git pull --rebase origin dev; then
  echo "rebase 完成"
else
  echo "rebase 過程發生衝突，請手動解決衝突後執行："
  echo "  git add <files>"
  echo "  git rebase --continue"
  echo "或若要放棄 rebase： git rebase --abort"
  exit 2
fi

# push
echo "推送到 origin/dev..."
if git push origin dev; then
  echo "推送成功。"
  echo "目前 HEAD: $(git rev-parse --short HEAD)"
  exit 0
else
  echo "推送失敗，可能需要設定上游或權限。若需要強制推送請確認後手動執行。"
  exit 3
fi
