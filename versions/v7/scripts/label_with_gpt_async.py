#!/usr/bin/env python3
"""
Concurrent GPT labeler for v9 unlabeled jobs.
Usage:
  OPENAI_API_KEY=... ./scripts/label_with_gpt_async.py --input /tmp/v9_unlabeled_all.jsonl --output /tmp/v9_gpt_labels_clean.jsonl --max-workers 12 --limit 0 --skip 0
"""
import argparse, asyncio, json, os
from pathlib import Path
import openai

SYSTEM = """You are a precise job-fit classifier. Output strict JSON with keys loc, arr, sen, tech, comp.
loc: IN_LONDON|REMOTE|UK_OTHER|OUTSIDE_UK|UNK
arr: REMOTE|HYBRID|IN_OFFICE|UNK
sen: LEVEL_3|LEVEL_2|LEVEL_1
tech: JSON array using NODE, REACT, JS_TS, AI_ML, OOS (use ["OOS"] if none)
comp: NO_GBP|UP_TO_ONLY|BELOW_45K|RANGE_45_54K|RANGE_55_74K|RANGE_75_99K|ABOVE_100K
Return only JSON, no text.
Rules:
- If any GBP/£ present, choose the closest GBP band (not NO_GBP).
- If Node/React/JavaScript/TypeScript appear, include those tech tokens; add AI_ML if ML/AI terms appear; use ["OOS"] only when none present.
- London -> IN_LONDON; other UK cities -> UK_OTHER; non-UK city/country -> OUTSIDE_UK.
- Remote keywords -> REMOTE; hybrid -> HYBRID; onsite/office -> IN_OFFICE.
- Senior/Lead/Principal/Staff -> LEVEL_3; Junior/Graduate/Entry -> LEVEL_1; else LEVEL_2."""

USER_TMPL = """Title: {title}
Location: {location}
Description: {jd}

{{"loc":"...","arr":"...","sen":"...","tech":[],"comp":"..."}}"""


def score_label(parsed):
    loc_score = {"IN_LONDON": 25, "UK_OTHER": 10, "REMOTE": 15, "OUTSIDE_UK": -50, "UNK": 0}.get(parsed.get("loc"), 0)
    role_score = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}.get(parsed.get("sen"), 0)
    comp_map = {
        "ABOVE_100K": 25,
        "RANGE_75_99K": 15,
        "RANGE_55_74K": 5,
        "RANGE_45_54K": 0,
        "UP_TO_ONLY": 0,
        "BELOW_45K": -30,
        "NO_GBP": 0,
    }
    comp_score = comp_map.get(parsed.get("comp"), 0)
    tech_score = 0
    t = parsed.get("tech") or []
    if "NODE" in t:
        tech_score += 10
    if "JS_TS" in t or "REACT" in t:
        tech_score += 5
    if "AI_ML" in t:
        tech_score += 10
    tech_score = min(25, tech_score)
    total = max(0, min(100, loc_score + role_score + comp_score + tech_score))
    if total >= 70:
        label = "good_fit"
    elif total >= 50:
        label = "maybe"
    else:
        label = "bad_fit"
    return total, label


async def classify(client, job, sem):
    prompt = USER_TMPL.format(
        title=job.get("title", ""),
        location=job.get("job_location", job.get("location", "")),
        jd=job.get("jd_text", ""),
    )
    async with sem:
        for _ in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
                txt = resp.choices[0].message.content
                parsed = json.loads(txt)
                score, label = score_label(parsed)
                job_out = {
                    **job,
                    "loc": parsed.get("loc"),
                    "arr": parsed.get("arr"),
                    "sen": parsed.get("sen"),
                    "tech": parsed.get("tech"),
                    "comp": parsed.get("comp"),
                    "score": score,
                    "label": label,
                    "source_file": "gpt_labeled",
                }
                return job_out
            except Exception:
                await asyncio.sleep(0.5)
                return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.max_workers)

    jobs = []
    with open(args.input) as f:
        for idx, line in enumerate(f, 1):
            if args.skip and idx <= args.skip:
                continue
            if args.limit and idx > args.limit:
                break
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    tasks = [asyncio.create_task(classify(client, job, sem)) for job in jobs]
    with out_path.open("w") as out:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res:
                out.write(json.dumps(res) + "\n")
                written += 1
    print(f"wrote {written} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
