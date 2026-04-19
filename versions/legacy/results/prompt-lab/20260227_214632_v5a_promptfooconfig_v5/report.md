# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v5a |
| **Config** | promptfooconfig_v5.yaml |
| **Timestamp** | 2026-02-27T21:50:19.122Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 50% |
| Parse Fail | 0% |
| Score MAE | 26 |
| Score Bias | +22 |
| Avg Latency | 22.4s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 3 | 1 | 0 | 0 |
| **maybe** | 1 | 2 | 0 | 0 |
| **bad_fit** | 0 | 3 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (90) | 70/90/+20 | 17.9s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 19.8s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (65) | 65/65/+0 | 22.6s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [MISS] | good_fit (75) | maybe (60) | 75/60/-15 | 16.3s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (85) | 90/85/-5 | 18.8s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (85) | 50/85/+35 | 25.2s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (65) | 0/65/+65 | 35.7s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | maybe (65) | 25/65/+40 | 22.4s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (85) | 80/85/+5 | 18.1s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [MISS] | bad_fit (0) | maybe (65) | 0/65/+65 | 26.6s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (5/10)

### Miss 1: 1785765898 | Senior Product Engineer @ Jenkins - Zboncak

- **Expected:** good_fit (score 75)
- **Predicted:** maybe (score 60)
- **Score error:** -15
- **Latency:** 16.3s

> **Model reasoning:** The candidate is a Senior Product Engineer, working remotely in the UK. The job description highlights specific technical skills like Rust and database optimization, but the provided information doesn't fully assess their suitability, thus a 'maybe' score is appropriate.

### Miss 2: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 85)
- **Score error:** +35
- **Latency:** 25.2s

> **Model reasoning:** The job title 'Backend Systems Developer' indicates a mid-level role. The tech stack includes Node.js, TypeScript, and PostgreSQL, earning 10, 5, and 10 points respectively. The location is Hybrid - Bristol, UK, earning 10 points. The salary range is £120,000 - £140,000, with a midpoint of £130,000, earning 25 points.

### Miss 3: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 65)
- **Score error:** +65
- **Latency:** 35.7s

> **Model reasoning:** The candidate is a Front End Software Developer located in the United States. The job description highlights experience with React, Redux, and frontend testing frameworks, aligning with the tech stack. However, the compensation range is in USD and doesn't provide a specific GBP figure, and the location is in the US, which is not a strong positive for a UK-based role. The role requires significant experience, suggesting a higher seniority level, but the scoring is capped at 25 for role/seniority.

### Miss 4: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** maybe (score 65)
- **Score error:** +40
- **Latency:** 22.4s

> **Model reasoning:** The candidate fits the Senior/Staff/Principal role and the tech stack requirements (React, TypeScript, Node.js, Next.js, Express, PostgreSQL). However, the salary is slightly above the upper limit (£150K) and the location is London, which is hybrid. A higher score would require a more precise salary match.

### Miss 5: 4375738047 | Full Stack Engineer @ Quikr

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 65)
- **Score error:** +65
- **Latency:** 26.6s

> **Model reasoning:** The job title is Full Stack Engineer, which scores 15 points. The job description mentions experience with Node.js, React, and RESTful APIs, earning 10+5 points. The location is Remote (United States), scoring 0 points. Compensation details are not provided, so it receives 0 points. Overall, it's a potentially good fit but lacks specific compensation information.

## Prompt Used

```
You are a strict job‑fit scorer for one candidate. Score ONLY what is explicitly stated. Do NOT infer missing info.

Use ONLY these fields:
- Job title: {{job_title}}
- Job location: {{job_location}}
- Job description: {{jd_text}}

Field rules:
- Role & Seniority: use ONLY job_title.
- Location: use ONLY job_location.
- Tech Stack: use ONLY jd_text (not the title).
- Compensation: use jd_text (or job_title if salary is explicitly stated there).

Scoring rubric — sum four categories (floor 0, cap 100):

1) Role & Seniority (max 25) — based on job_title only
+25: Title explicitly includes Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, or Founding
+15: Title explicitly includes Full Stack, Mid‑Level/Mid Level/Midlevel, Software Engineer II / SWE II / Engineer II
0: Otherwise

2) Tech Stack (max 25, points stack; cap at 25) — based on jd_text only
+10: Node.js explicitly required or listed in core stack/requirements
+5: JavaScript or TypeScript explicitly required
+10: AI/ML/LLM experience explicitly required (not just company uses AI)

3) Location & Work Arrangement (max 25) — based on job_location only
+25: Remote (UK/global) OR hybrid/on‑site in London
+10: Hybrid/on‑site in UK outside London
0: Location missing or unclear
-50: Location explicitly outside the UK

4) Compensation (max 25) — base salary in GBP only
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

```
