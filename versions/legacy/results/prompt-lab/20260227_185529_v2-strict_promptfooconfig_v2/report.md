# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v2-strict |
| **Config** | promptfooconfig_v2.yaml |
| **Timestamp** | 2026-02-27T19:03:45.577Z |
| **Jobs** | 10 |
| **Status** | FAIL* |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 50% |
| Parse Fail | 25% |
| Score MAE | 6.7 |
| Score Bias | +3.3 |
| Avg Latency | 123.3s |
| Tests Run | 4 |

**Fail reasons:** TIMEOUT on job 4 (>420s)

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 0 | 1 | 0 | 1 |
| **maybe** | 0 | 2 | 0 | 0 |
| **bad_fit** | 0 | 0 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (65) | 70/65/-5 | 19.7s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (55) | 50/55/+5 | 31.1s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (75) | 65/75/+10 | 22.1s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [PARSE] | good_fit (75) | — (—) | 75/—/— | 420.2s | 1785765898 | Senior Product Engineer @ Jenkins ... |

## Misses (2/4)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 65)
- **Score error:** -5
- **Latency:** 19.7s

> **Model reasoning:** Role & Seniority: 15 points (Lead DevOps Architect implies senior level). Tech Stack & Domain: 15 points (Python, PyTorch, LLMs). Location & Work Arrangement: 10 points (Hybrid in Manchester, UK). Compensation: 15 points (Salary £100k - £115k).

### Miss 2: 1785765898 | Senior Product Engineer @ Jenkins - Zboncak

- **Expected:** good_fit (score 75)
- **Predicted:** PARSE FAIL (score —)
- **Latency:** 420.2s

> **Raw output:** `TIMEOUT`

## Prompt Used

```
You are a strict job-fit scorer for one candidate. Score by the rubric ONLY — do not guess or inflate.

Candidate:
- UK-based senior/staff backend engineer (8+ yrs)
- Core stack: TypeScript, Node.js, Python, SQL, cloud/distributed systems
- Interested in AI/ML/LLM roles; prefers product teams over agency/consulting
- Target level: Senior, Staff, Principal, or Lead IC (NOT management-only)
- Location: remote (UK or global) or hybrid in London only
- Salary: £75k+ GBP base

IMPORTANT RULES:
- Score ONLY what the job description explicitly states. Do not assume or infer missing information.
- If salary is not listed, award 0 pts for compensation — do NOT guess.
- If location is unclear or missing, award 0 pts for location.
- Recruiter posts (third-party ads listing multiple roles or vague descriptions) should be scored as bad_fit unless they contain enough concrete detail to evaluate.
- The label MUST match the score range exactly. No exceptions.

Scoring rubric — sum these four categories (floor 0, cap 100):

1. Role & Seniority (max 25 pts):
   +25: Senior, Staff, Principal, Lead, or Tech Lead Engineer
   +15: Full Stack, Mid-Level, or Software Engineer II
   0: Junior, management-only (EM), or non-engineering roles (DevOps, QA, Frontend-only)

2. Tech Stack & Domain (max 25 pts, points stack, cap at 25):
   +10: Node.js explicitly required or listed
   +5: JavaScript or TypeScript explicitly required or listed
   +10: AI, ML, or LLM experience explicitly mentioned
   0: If the primary stack is unrelated (e.g., Java, C#, Ruby only)

3. Location & Work Arrangement (max 25 pts):
   +25: Fully remote (UK-based or global) OR hybrid in London
   +10: Hybrid or on-site in UK outside London
   0: Location unknown or not stated
   -50: On-site or hybrid outside the UK (this alone makes the job bad_fit)

4. Compensation (max 25 pts):
   +25: Base salary £100k+ GBP
   +15: Base salary £75k–£99k GBP
   +5: Base salary £55k–£74k GBP
   0: Salary unknown, not listed, or given in non-GBP without clear conversion
   -30: Base salary below £45k GBP

Label mapping (MUST match score):
- good_fit: score 70–100
- maybe: score 50–69
- bad_fit: score 0–49

Respond with ONLY a JSON object. No markdown, no explanation outside the JSON.
{"label":"good_fit|maybe|bad_fit","score":<integer 0-100>,"reasoning":"<1-2 sentences explaining category scores>"}

Job description:
{{jd_text}}

```
