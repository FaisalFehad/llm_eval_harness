# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v5 |
| **Config** | promptfooconfig_v5.yaml |
| **Timestamp** | 2026-02-27T21:37:46.983Z |
| **Jobs** | 10 |
| **Status** | FAIL* |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 71.4% |
| Parse Fail | 14.3% |
| Score MAE | 6.7 |
| Score Bias | +1.7 |
| Avg Latency | 75.8s |
| Tests Run | 7 |

**Fail reasons:** TIMEOUT on job 7 (>420s)

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 2 | 1 | 0 | 0 |
| **maybe** | 0 | 3 | 0 | 0 |
| **bad_fit** | 0 | 0 | 0 | 1 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (65) | 70/65/-5 | 22.7s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (40) | 50/40/-10 | 19.8s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (65) | 65/65/+0 | 18.6s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (85) | 75/85/+10 | 15.7s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 15.8s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [OK] | maybe (50) | maybe (65) | 50/65/+15 | 17.9s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [PARSE] | bad_fit (0) | — (—) | 0/—/— | 420.1s | 4374501376 | Front End Software Develper @ Oracle |

## Misses (2/7)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 65)
- **Score error:** -5
- **Latency:** 22.7s

> **Model reasoning:** The role is for a Lead DevOps Architect, which aligns with the candidate's interest in senior roles. However, the tech stack primarily focuses on Python and LLMs, not Node/React/TypeScript, and the location is Manchester, UK, not London, impacting the overall fit.

### Miss 2: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** PARSE FAIL (score —)
- **Latency:** 420.1s

> **Raw output:** `TIMEOUT`

## Prompt Used

```
You are a strict job‑fit scorer for one candidate. Score ONLY what the job description explicitly states. Do NOT infer missing info.

Candidate context (for scoring only):
- UK‑based senior/staff engineer interested in Node/React/TypeScript and AI/ML roles.
- Prefers remote (UK/global) or hybrid in London.
- Salary target £75k+.

Scoring rubric — sum four categories (floor 0, cap 100):

1) Role & Seniority (max 25)
+25: Title explicitly includes Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, or Founding
+15: Title explicitly includes Full Stack, Mid‑Level/Mid Level/Midlevel, Software Engineer II / SWE II / Engineer II
0: Otherwise

2) Tech Stack (max 25, points stack; cap at 25)
+10: Node.js OR React is explicitly required or listed in core stack/requirements
+5: JavaScript or TypeScript is explicitly required
+10: AI/ML/LLM experience is explicitly required (not just company uses AI)

3) Location & Work Arrangement (max 25)
+25: Fully remote (UK/global) OR remote (UK/England/Scotland/etc) OR hybrid/on‑site in London
+10: Hybrid/on‑site in UK outside London
0: Location not stated or unclear
-50: Location explicitly outside the UK

4) Compensation (max 25)
- Only count base salary in GBP explicitly stated as salary/compensation.
- Ignore bonuses, equity, benefits, allowances, budgets, and non‑GBP amounts.
- If a range is given, use the midpoint to choose the bracket.
+25: midpoint ≥ £100k
+15: midpoint £75k–£99k
+5: midpoint £55k–£74k
-30: midpoint < £45k
0: salary not stated

Label mapping (must match score):
- good_fit: 70–100
- maybe: 50–69
- bad_fit: 0–49

Return ONLY JSON:
{"label":"good_fit|maybe|bad_fit","score":0-100,"reasoning":"1-2 sentences"}

Job description:
{{jd_text}}

```
