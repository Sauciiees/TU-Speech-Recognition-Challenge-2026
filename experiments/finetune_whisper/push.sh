#!/usr/bin/env bash
# Push finetune_whisper.py to its own Kaggle notebook for one model.
# Usage: bash push.sh typhoon-ai/typhoon-whisper-large-v3 dsi443-lora-typhoon-large [EXPERIMENT,...]
set -euo pipefail
model="$1"
slug="$2"
experiments="${3:-}"
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
sed "s#^MODEL_ID = .*#MODEL_ID = \"$model\"#" "$here/finetune_whisper.py" > "$tmp/finetune_whisper.py"
if [ -n "$experiments" ]; then
  list=$(echo "$experiments" | sed 's/[^,]*/"&"/g; s/,/, /g')
  sed -i "s#^DEFAULT_EXPERIMENTS = .*#DEFAULT_EXPERIMENTS = [$list]#" "$tmp/finetune_whisper.py"
  grep -n '^DEFAULT_EXPERIMENTS' "$tmp/finetune_whisper.py"
fi
cat > "$tmp/kernel-metadata.json" <<EOF
{
  "id": "sauciiees/$slug",
  "title": "$slug",
  "code_file": "finetune_whisper.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["dsi-443-tu-speech-recognition-challenge-2026"],
  "dataset_sources": [],
  "kernel_sources": []
}
EOF
grep -n '^MODEL_ID' "$tmp/finetune_whisper.py"
python -m kaggle kernels push -p "$tmp"
