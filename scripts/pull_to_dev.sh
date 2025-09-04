cd /root/learnCLIP
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "<unknown>")
echo "Current branch: $BRANCH"

echo "--- git status (porcelain) ---"
git status --porcelain || true

echo "--- fetching from remote ---"
git fetch --all --prune || true

echo "--- resetting to origin/$BRANCH (this will discard local changes) ---"
git reset --hard origin/$BRANCH || true

echo "--- removing untracked files/directories ---"
git clean -fd || true

echo "--- latest commit now ---"
git log -1 --oneline || true

echo "--- final git status (porcelain) ---"
git status --porcelain || true