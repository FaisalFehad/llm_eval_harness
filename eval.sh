#!/bin/bash
#
# eval.sh — quick CLI wrapper for promptfoo eval pipeline
#
# Usage: ./eval.sh help

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="versions/v15/configs/promptfoo.yaml"
WRAP="versions/v15/configs/wrap_prompt.cjs"
OUTPUT="versions/v15/eval_results/master.json"

# Langfuse env vars (auto-set for hooks)
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-eval-harness-local}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-eval-harness-local}"
export LANGFUSE_HOST="${LANGFUSE_HOST:-http://localhost:3000}"

# ── Helpers ──────────────────────────────────────────────────────
prompt_flag() {
    local name="$1"
    case "$name" in
        production|prod) echo "file://$WRAP:v15_production" ;;
        fix1)            echo "file://$WRAP:v15_fix1" ;;
        fix2)            echo "file://$WRAP:v15_fix2" ;;
        fix3)            echo "file://$WRAP:v15_fix3" ;;
        fix4)            echo "file://$WRAP:v15_fix4" ;;
        fix5)            echo "file://$WRAP:v15_fix5" ;;
        base)            echo "file://$WRAP:v15_base" ;;
        *)               echo "$name" ;;
    esac
}

# ── Commands ─────────────────────────────────────────────────────
cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in

    run)
        if [ $# -gt 0 ]; then
            p=$(prompt_flag "$1")
            echo "Running prompt: $1"
            npx promptfoo eval -c "$CONFIG" -p "$p"
        else
            echo "Running all prompts..."
            npx promptfoo eval -c "$CONFIG"
        fi
        ;;

    compare)
        if [ $# -lt 2 ]; then
            echo "Usage: ./eval.sh compare <prompt1> <prompt2>"
            echo "Example: ./eval.sh compare fix3 fix4"
            exit 1
        fi
        p1=$(prompt_flag "$1")
        p2=$(prompt_flag "$2")
        echo "Comparing: $1 vs $2"
        npx promptfoo eval -c "$CONFIG" -p "$p1" -p "$p2"
        ;;

    smoke)
        n="${1:-10}"
        echo "Smoke test: first $n jobs..."
        npx promptfoo eval -c "$CONFIG" --filter-first-n "$n"
        ;;

    failing)
        # --filter-failing needs an eval ID or output file path
        src="${1:-$OUTPUT}"
        echo "Re-running failing tests from: $src"
        npx promptfoo eval -c "$CONFIG" --filter-failing "$src"
        ;;

    subset)
        if [ $# -lt 1 ]; then
            echo "Usage: ./eval.sh subset <golden_label>"
            echo "Example: ./eval.sh subset good_fit"
            exit 1
        fi
        echo "Running subset: golden_label=$1"
        npx promptfoo eval -c "$CONFIG" --filter-metadata "golden_label=$1"
        ;;

    view)
        echo "Opening promptfoo results UI..."
        npx promptfoo view
        ;;

    langfuse)
        echo "Opening Langfuse UI..."
        open "http://localhost:3000"
        ;;

    runs)
        npx promptfoo list evals
        ;;

    watch)
        echo "Watch mode — auto-reruns on config/prompt changes..."
        npx promptfoo eval -c "$CONFIG" -w
        ;;

    server)
        port="${1:-8000}"
        model="${MLX_MODEL:-$HOME/MLX Models/qwen3_4B_v15_oQ6}"
        echo "Starting MLX server on port $port with model: $model"
        .venv/bin/python3 -m mlx_lm server \
            --model "$model" \
            --port "$port" \
            --host 0.0.0.0 \
            --max-tokens 500
        ;;

    server-stop)
        echo "Stopping MLX server..."
        pkill -f "mlx_lm server" 2>/dev/null && echo "Stopped." || echo "No server running."
        ;;

    langfuse-up)
        echo "Starting Langfuse..."
        cd langfuse && docker compose up -d
        echo "Langfuse UI: http://localhost:3000"
        echo "Login: faisal@local.dev / changeme123"
        ;;

    langfuse-down)
        echo "Stopping Langfuse..."
        cd langfuse && docker compose down
        ;;

    help|*)
        cat << 'EOF'
eval.sh — promptfoo eval pipeline CLI

  EVAL COMMANDS:
    run [prompt]         Run eval (all prompts, or specify: production, fix4, etc.)
    compare <p1> <p2>    Compare two prompts side-by-side
    smoke [n]            Quick test with first n jobs (default: 10)
    failing [src]        Re-run failing tests (from last output or specify eval ID/path)
    subset <label>       Run only tests matching golden_label (good_fit, maybe, bad_fit)
    watch                Auto-rerun on config/prompt file changes

  VIEW COMMANDS:
    view                 Open promptfoo results UI (browser)
    langfuse             Open Langfuse tracing UI (browser)
    runs                 List past eval runs

  INFRA COMMANDS:
    server [port]        Start MLX model server (default port 8000)
    server-stop          Stop MLX model server
    langfuse-up          Start Langfuse Docker containers
    langfuse-down        Stop Langfuse Docker containers

  PROMPT SHORTCUTS:
    production, prod, fix1, fix2, fix3, fix4, fix5, base

  EXAMPLES:
    ./eval.sh server                      # start model
    ./eval.sh run production              # eval production prompt
    ./eval.sh compare fix3 fix4           # A/B test
    ./eval.sh smoke 5                     # quick 5-job test
    ./eval.sh failing                     # re-run failures from last run
    ./eval.sh view                        # see results
EOF
        ;;
esac
