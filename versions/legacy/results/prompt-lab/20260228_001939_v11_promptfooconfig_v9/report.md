# Prompt Lab Report

| | |
|---|---|
| **Model** | qwen3-4b-instruct-2507 |
| **Tag** | v11 |
| **Config** | qwen3-4b-instruct-2507/promptfooconfig_v9.yaml |
| **Timestamp** | 2026-02-28T00:24:30.258Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 80% |
| Parse Fail | 0% |
| Score MAE | 12.5 |
| Score Bias | +12.5 |
| Avg Latency | 28.8s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 4 | 0 | 0 | 0 |
| **maybe** | 1 | 2 | 0 | 0 |
| **bad_fit** | 1 | 0 | 2 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (75) | 70/75/+5 | 47.9s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 17.3s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (65) | 65/65/+0 | 25.3s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (85) | 75/85/+10 | 46.0s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 22.4s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (75) | 50/75/+25 | 40.3s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 26.2s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (90) | 25/90/+65 | 21.1s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (90) | 80/90/+10 | 21.7s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 20.2s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (2/10)

### Miss 1: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 75)
- **Score error:** +25
- **Latency:** 40.3s

> **Model reasoning:** Location is hybrid in Bristol, UK (10 points). Role contains 'Backend Systems Developer' — no explicit Senior/Lead keywords, so role score is 0. However, 'Systems Developer' is not in the list of senior keywords, so role score should be 0. Correction: title does not contain Senior, Lead, etc. Only 'Backend' and 'Systems' — none of the senior keywords. So role score must be 0. Re-evaluate role: title is 'Backend Systems Developer' — contains no Senior, Lead, etc. So role = 0. Tech: Node.js and TypeScript explicitly listed → +10 (Node.js) +5 (TypeScript) = 15. Compensation: £120,000-£140,000 → midpoint = £130,000 → >= £100,000 → +25. Location: hybrid in Bristol (UK city not London) → +10. Role: no senior keyword → 0. So correct scoring: loc=10, role=0, tech=15, comp=25. Total = 10+0+15+25 = 50 → maybe_fit. But wait: role title does not contain any of the senior keywords. So role = 0. Final: loc=10, role=0, tech=15, comp=25 → total=50. Label: maybe_fit.

### Miss 2: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 90)
- **Score error:** +65
- **Latency:** 21.1s

> **Model reasoning:** Location is hybrid in London Area, UK → +25. Title contains 'Senior' → +25. JD requires Node.js (+10), TypeScript (+5) → total tech 15. Base salary 'up to £170K' → midpoint £170k → ≥£100k → +25. Total: 25+25+15+25 = 90.

## Prompt Used

```
You are a mechanical job-fit scorer. Follow every rule exactly. Score ONLY what is explicitly written. Do NOT infer or assume.

Inputs:
- Job title: {{job_title}}
- Job location: {{job_location}}
- Job description: {{jd_text}}

===== STEP 1: LOCATION (score this FIRST) =====

Use ONLY the job_location field. Ignore any locations mentioned inside jd_text.

  -50 if job_location is outside the UK (e.g. United States, US, USA, India, Germany, California, Bangalore, Sydney, Remote (US), Seattle, Canada)
  +25 if Remote (UK/Global/Worldwide) OR hybrid/on-site in London
  +10 if hybrid/on-site in UK city that is NOT London (e.g. Manchester, Bristol, Edinburgh)
  0   if location is missing or unclear

===== STEP 2: ROLE & SENIORITY (max 25) =====

Search job_title for these EXACT keywords (case-insensitive):
  +25 if title contains ANY of: Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding
  +15 if title contains ANY of: Full Stack, Full-Stack, Fullstack, Mid-Level, Mid Level, Midlevel, Software Engineer II, Engineer II, SWE II
  0   if NONE of the above keywords appear

Do NOT give points for: Engineer, Developer, Backend, Frontend, Architect, Manager, Product, or any word not listed above.

===== STEP 3: TECH STACK (max 25, points stack, cap 25) =====

Search jd_text requirements/core stack sections ONLY:
  +10 if Node.js or NodeJS is listed as required/core
  +5  if JavaScript or TypeScript is listed as required/core
  +10 if AI/ML/LLM experience is explicitly REQUIRED (e.g. "experience building ML models", "LLM integrations required")

Do NOT award AI points if AI is only in company description, mission, or "nice to have."
Do NOT award points for unlisted tech: React, PostgreSQL, Python, Go, Rust, Java = 0 points each.

===== STEP 4: COMPENSATION (max 25) =====

From jd_text ONLY. Count ONLY base salary explicitly in GBP (£).
Ignore: bonuses, equity, benefits, USD ($), EUR, and all non-GBP amounts.

Converting "k" notation: £100k = £100,000. £80k-£120k = £80,000-£120,000.
- If a range is given (e.g. £80,000-£120,000), use the midpoint.
- "Up to £X" with no lower bound = salary not stated, score 0.

  +25 if midpoint >= £100,000
  +15 if midpoint £75,000-£99,999
  +5  if midpoint £55,000-£74,999
  -30 if midpoint < £45,000
  0   if no GBP salary stated or "up to" with no lower bound

===== WORKED EXAMPLES =====

Example A (good_fit):
Title: "Senior Software Engineer" | Location: "London Area, United Kingdom (Hybrid)" | JD requires Node.js, TypeScript, salary £90,000-£110,000.
→ loc: 25 (London, UK)
→ role: 25 (Senior)
→ tech: 15 (Node.js +10, TypeScript +5)
→ comp: 25 (midpoint £100,000)
→ total: 25+25+15+25 = 90 → good_fit
{"loc":25,"role":25,"tech":15,"comp":25,"score":90,"label":"good_fit","reasoning":"Senior role in London with Node.js/TS and £100k midpoint salary."}

Example B (bad_fit):
Title: "Full Stack Engineer" | Location: "United States" | JD mentions React, Node.js, TypeScript, competitive salary.
→ loc: -50 (United States = outside UK)
→ role: 15 (Full Stack)
→ tech: 15 (Node.js +10, TypeScript +5; React = 0)
→ comp: 0 (no GBP salary)
→ total: max(0, -50+15+15+0) = 0 → bad_fit
{"loc":-50,"role":15,"tech":15,"comp":0,"score":0,"label":"bad_fit","reasoning":"US location (-50) makes total negative, floored to 0."}

===== OUTPUT FORMAT =====

Score each category independently, then total = max(0, min(100, loc+role+tech+comp)).

Label MUST match total:
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

Return ONLY this JSON:
{"loc":0,"role":0,"tech":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"brief"}

```
