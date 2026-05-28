#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/rodriguem-lab/econometrics-projects-13.git"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This folder is not a git repository."
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git branch -M main
git push -u origin main
