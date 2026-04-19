#!/bin/bash
# Regenerates .env from macOS Keychain + config defaults
# Usage: ./gen-env.sh
set -euo pipefail

kc() { security find-generic-password -a "$1" -s "$2" -w 2>/dev/null || echo "MISSING_FROM_KEYCHAIN"; }

cat > "$(dirname "$0")/.env" <<EOF
# API Keys — this file is gitignored
OPENAI_API_KEY=$(kc ai_eval_harness OPENAI_API_KEY)
EOF

echo "✓ .env generated from Keychain for ai_eval_harness"
