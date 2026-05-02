#!/usr/bin/env bash
# Download ScanNet monocular subset.
#
# ScanNet requires registering a license. After acceptance, drop the bundled
# tar at data/_raw/scannet_v2.tgz and run this script to extract + index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/scannet"
RAW="$ROOT/data/_raw/scannet_v2.tgz"

mkdir -p "$DATA"
if [[ ! -f "$RAW" ]]; then
  echo "Place ScanNet v2 archive at $RAW first."
  exit 1
fi

tar -xf "$RAW" -C "$DATA"
echo "Done."
