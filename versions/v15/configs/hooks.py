"""
Promptfoo extension hooks — runs after each eval to:
1. Print a summary table (hybrid accuracy, per-field, parse failures)
2. Push scores to Langfuse if available

Registered in promptfooconfig via:
  extensions:
    - file://v15_hooks.py
"""
import os
from datetime import datetime


# ── Langfuse client (optional — graceful if not running) ─────────
_langfuse = None

def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-eval-harness-local"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-eval-harness-local"),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
        return _langfuse
    except Exception:
        _langfuse = False  # don't retry
        return False


def after_all(hook_name, context):
    """Runs after the full eval suite completes."""
    results = context.get("results", [])
    suite = context.get("suite", {})

    if not results:
        print("\n[hooks] No results to summarize.")
        return

    # ── Aggregate metrics ────────────────────────────────────────
    total = len(results)
    passed = 0
    parse_fails = 0
    field_correct = {"loc": 0, "arr": 0, "sen": 0, "tech": 0, "comp": 0}
    field_total = {"loc": 0, "arr": 0, "sen": 0, "tech": 0, "comp": 0}
    errors = []

    for r in results:
        grading = r.get("gradingResult") or {}
        components = grading.get("componentResults") or []

        # Check if this was a parse failure (score 0, reason contains PARSE FAIL)
        reason = grading.get("reason") or ""
        if "PARSE FAIL" in reason:
            parse_fails += 1
            continue

        if grading.get("pass", False):
            passed += 1

        # Per-field accuracy from componentResults
        for component in components:
            comp_reason = component.get("reason", "")
            for field in field_correct:
                if comp_reason.startswith(f"{field}:"):
                    field_total[field] += 1
                    if component.get("pass", False):
                        field_correct[field] += 1
                    else:
                        # Collect error details
                        test_meta = r.get("vars", {})
                        job_idx = test_meta.get("metadata", {}).get("job_index", "?") if isinstance(test_meta.get("metadata"), dict) else "?"
                        errors.append({
                            "job": job_idx,
                            "field": field,
                            "detail": comp_reason,
                        })

    scorable = total - parse_fails
    hybrid_pct = (passed / total * 100) if total > 0 else 0
    model_pct = (passed / scorable * 100) if scorable > 0 else 0

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVAL SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  Total tests:     {total}")
    print(f"  Parse failures:  {parse_fails}")
    print(f"  Hybrid accuracy: {passed}/{total} ({hybrid_pct:.1f}%)")
    if scorable != total:
        print(f"  Model-only:      {passed}/{scorable} ({model_pct:.1f}%)")
    print("-" * 60)

    # Per-field
    print(f"  {'Field':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for f in ["loc", "arr", "sen", "tech", "comp"]:
        ft = field_total[f]
        fc = field_correct[f]
        pct = (fc / ft * 100) if ft > 0 else 0
        print(f"  {f:<8} {fc:>8} {ft:>8} {pct:>9.1f}%")
    print("=" * 60)

    # Top errors
    if errors:
        print(f"\n  Errors ({len(errors)} total):")
        for e in errors[:15]:
            print(f"    Job {e['job']}: {e['detail']}")
        if len(errors) > 15:
            print(f"    ... and {len(errors) - 15} more")
        print()

    # ── Push to Langfuse ─────────────────────────────────────────
    lf = _get_langfuse()
    if not lf:
        print("[hooks] Langfuse not available — skipping score upload.")
        return

    try:
        run_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        description = suite.get("description", "V15 eval")

        # Create a trace for this eval run (return value unused — scores reference run_id directly)
        lf.trace(
            name="promptfoo-eval",
            id=run_id,
            metadata={
                "description": description,
                "total_tests": total,
                "parse_failures": parse_fails,
                "provider": str(suite.get("providers", ["unknown"])),
            },
        )

        # Attach scores
        lf.score(
            trace_id=run_id,
            name="hybrid_accuracy",
            value=hybrid_pct / 100,
            data_type="NUMERIC",
            comment=f"{passed}/{total}",
        )
        lf.score(
            trace_id=run_id,
            name="parse_failures",
            value=parse_fails,
            data_type="NUMERIC",
            comment=f"{parse_fails}/{total} failed to parse",
        )

        for f in ["loc", "arr", "sen", "tech", "comp"]:
            ft = field_total[f]
            fc = field_correct[f]
            if ft > 0:
                lf.score(
                    trace_id=run_id,
                    name=f"{f}_accuracy",
                    value=fc / ft,
                    data_type="NUMERIC",
                    comment=f"{fc}/{ft}",
                )

        lf.flush()
        print(f"[hooks] Scores pushed to Langfuse (trace: {run_id})")
        print(f"[hooks] View at http://localhost:3000")
    except Exception as e:
        print(f"[hooks] Langfuse error (non-fatal): {e}")
