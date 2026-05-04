#!/bin/bash
# Resume V16 generation with 4 parallel backends
# 2x Ollama gemma4:31b-cloud (faster) + 2x OMLX gemma-4-31B-it-oQ4

set -e

INPUT="versions/v16/data/v15_jobs_extracted.jsonl"
OUTDIR="versions/v16/data/full_relabel"
PROMPT="versions/v16/prompts/teacher.txt"
API_KEY="fbe900938b6dedf7eb6994a03054ebbe1654b7d6788c35fdccde7568c8c70e76f300f3faebec1ff7e7066b45a07b7920d406a445682c9019ff71f9cae4d07c17ee473c670c2442d44b04d1b3f4f74f650453d0e0b7743567a525cd3c34f7df5553c64f96fb940b12ab15a8b5d094b5a53dbf7daf9a81fef711b55f7ff2b07c35"

mkdir -p "$OUTDIR"

# Collect all already-done indices
echo "Finding completed jobs..."
python3 -c "
import json, glob
from pathlib import Path
done = set()
for f in glob.glob('$OUTDIR/batch_*.jsonl') + glob.glob('versions/v16/data/ollama_100/batch_*.jsonl'):
    if Path(f).exists():
        try:
            with open(f) as fd:
                for line in fd:
                    d = json.loads(line.strip())
                    done.add(d.get('index', d.get('job_id')))
        except:
            pass
with open('/tmp/done_indices.txt','w') as f:
    for idx in done:
        f.write(str(idx)+'\n')
print(f'Done: {len(done)} jobs')
"

# Extract remaining jobs
python3 -c "
import json, sys
with open('/tmp/done_indices.txt') as f:
    done = set(int(l.strip()) for l in f if l.strip())
with open('$INPUT') as f:
    jobs = [json.loads(l) for l in f]
remaining = [j for j in jobs if j.get('index') not in done]
print(f'Remaining: {len(remaining)} / {len(jobs)}', file=sys.stderr)

# Slice into 4 (distribute heavier jobs to Ollama)
n = len(remaining) // 4
slices = [remaining[:n], remaining[n:2*n], remaining[2*n:3*n], remaining[3*n:]]
for i, sl in enumerate(slices):
    with open(f'/tmp/remain_{i}.jsonl','w') as f:
        for j in sl:
            f.write(json.dumps(j)+'\n')
"

echo "Launching 4 parallel processes..."

# Ollama batch 0  (fast)
python3 finetune/relabel_for_drift_v16.py \
  --input /tmp/remain_0.jsonl \
  --output "$OUTDIR/batch_ollama_0.jsonl" \
  --prompt "$PROMPT" \
  --base-url http://localhost:11434/v1 \
  --api-key "" \
  --model gemma4:31b-cloud \
  > "$OUTDIR/log_ollama_0.txt" &
echo "[$!] ollama_0 started"

# Ollama batch 1  (fast)
python3 finetune/relabel_for_drift_v16.py \
  --input /tmp/remain_1.jsonl \
  --output "$OUTDIR/batch_ollama_1.jsonl" \
  --prompt "$PROMPT" \
  --base-url http://localhost:11434/v1 \
  --api-key "" \
  --model gemma4:31b-cloud \
  > "$OUTDIR/log_ollama_1.txt" &
echo "[$!] olla_ma_1 started"

# OMLX batch 0  (slower)
python3 finetune/relabel_for_drift_v16.py \
  --input /tmp/remain_2.jsonl \
  --output "$OUTDIR/batch_omlx_0.jsonl" \
  --prompt "$PROMPT" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key "$API_KEY" \
  --model gemma-4-31B-it-oQ4 \
  > "$OUTDIR/log_omlx_0.txt" &
echo "[$!] omlx_0 started"

# OMLX batch 1  (slower)
python3 finetune/relabel_for_drift_v16.py \
  --input /tmp/remain_3.jsonl \
  --output "$OUTDIR/batch_omlx_1.jsonl" \
  --prompt "$PROMPT" \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key "$API_KEY" \
  --model gemma-4-31B-it-oQ4 \
  > "$OUTDIR/log_omlx_1.txt" &
echo "[$!] omlx_1 started"

echo ""
echo "All 4 processes launched. Check progress with:"
echo "  tail -f $OUTDIR/log_ollama_*.txt $OUTDIR/log_omlx_*.txt"
echo ""
echo "Wait for all to finish..."
wait
echo "All done."
