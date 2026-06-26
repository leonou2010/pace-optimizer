#!/bin/bash
# Fetch the large GPT-2 pretraining sweep data used by the pretraining figures.
#
# The figures that ship their data in-repo (Fig 1, Fig 3, clip-frac, Tulu, the
# fine-tune appendix figures) work without this. The dense nanoGPT pretraining
# sweep (~180 MB) is hosted as a GitHub release asset to keep the git repo small.
#
# Usage:  bash data/download_extended.sh
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/extended"
URL="${PACE_EXTENDED_URL:-https://github.com/REPLACE_ME/pace/releases/download/v0.1/pace_extended_data.tar.gz}"

mkdir -p "$DEST"
echo "Downloading extended pretraining data to $DEST ..."
if command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$DEST/pace_extended_data.tar.gz"
else
    wget -O "$DEST/pace_extended_data.tar.gz" "$URL"
fi
tar -xzf "$DEST/pace_extended_data.tar.gz" -C "$DEST"
rm -f "$DEST/pace_extended_data.tar.gz"
echo "Done. Extended data in $DEST"
