#!/usr/bin/env bash
# Download the OPEN Replica dataset (no registration), preprocessed for monocular
# SLAM / 3DGS exactly as MonoGS and NICE-SLAM consume it. ScanNet is license-gated
# and is intentionally NOT used here; Replica is the open substitute.
#
# Source: the NICE-SLAM preprocessed Replica mirror (ETH CVG), which MonoGS's own
# README points at. ~12 GB, 8 scenes (room0-2, office0-4) with posed RGB-D frames.
#
# Usage:
#   bash scripts/download_replica.sh                 # -> data/replica/
#   REPLICA_DEST=/scratch.global/$USER/replica bash scripts/download_replica.sh   # on MSI
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPLICA_DEST:-$ROOT/data/replica}"
URL="https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
mkdir -p "$DEST"

echo "[replica-dl] dest=$DEST"
ZIP="$DEST/Replica.zip"
if [[ ! -f "$ZIP" ]]; then
  echo "[replica-dl] fetching $URL (~12 GB) ..."
  # MSI login node has a 15-min CPU cap; run this inside srun/sbatch or on a data-mover.
  if command -v wget >/dev/null; then
    wget -c -O "$ZIP" "$URL"
  else
    curl -fL -C - -o "$ZIP" "$URL"
  fi
fi

echo "[replica-dl] extracting ..."
unzip -q -o "$ZIP" -d "$DEST"
echo "[replica-dl] done. Scenes under: $DEST/Replica/"
echo "[replica-dl] point the geosam config dataset.root at $DEST/Replica (replaces the ScanNet/random loader)."
