# data/v12/ — Shared Test Set

This directory is intentionally empty. The audited test set lives at:

**`versions/v12/data/v12_original/test_labeled_audited.jsonl`** (239 jobs, chmod 444)

## Why here?

Per `versions/README.md`, all versions share a single golden test set for reproducibility. The canonical copy lives in `versions/v12/` (the version that established it).

## Usage

```bash
# All evals reference this file
harness eval run --version v15 \
  --test-file versions/v12/data/v12_original/test_labeled_audited.jsonl

# Or use the registry default (all pipelines point here)
harness eval run --version v15
```

## File contents

| File | Jobs | Purpose |
|------|------|---------|
| `test_labeled_audited.jsonl` | 239 | Golden test set — audited labels, used for all V12+ evals |

## Related

- `versions/v12/data/v12_original/` — Full V12 data directory with train/valid splits
- `versions/README.md` — Per-version layout spec
