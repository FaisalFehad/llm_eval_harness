# Prompt Lab Report

| | |
|---|---|
| **Model** | qwen2.5-7b-instruct |
| **Tag** | v10 |
| **Config** | qwen2.5-7b-instruct/promptfooconfig_v10.yaml |
| **Timestamp** | 2026-02-28T01:08:39.799Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 70% |
| Parse Fail | 0% |
| Score MAE | 12 |
| Score Bias | +6 |
| Avg Latency | 23.5s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 3 | 1 | 0 | 0 |
| **maybe** | 0 | 3 | 0 | 0 |
| **bad_fit** | 1 | 1 | 1 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (60) | 70/60/-10 | 22.0s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (50) | 50/50/+0 | 22.5s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (50) | 65/50/-15 | 25.0s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (75) | 75/75/+0 | 19.6s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (85) | 90/85/-5 | 22.7s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 21.5s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (15) | 0/15/+15 | 33.6s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (85) | 25/85/+60 | 23.1s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (85) | 80/85/+5 | 22.4s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 23.2s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (3/10)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 60)
- **Score error:** -10
- **Latency:** 22.0s

> **Model reasoning:** Hybrid in UK city (Manchester) + Lead role + £100,000-£115,000 midpoint salary.

### Miss 2: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 15)
- **Score error:** +15
- **Latency:** 33.6s

> **Model reasoning:** US location (-50) and JavaScript/TypeScript (10) in tech stack, with a midpoint salary of £78,950 (midpoint of £79,800-£178,100).

### Miss 3: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 85)
- **Score error:** +60
- **Latency:** 23.1s

> **Model reasoning:** Senior role in London with Node.js/TS and £100k midpoint salary.

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

From jd_text ONLY. Count ONLY base salary explicitly stated as a GBP (£) range with BOTH a lower bound and upper bound.
Ignore: bonuses, equity, benefits, USD ($), EUR, and all non-GBP amounts.

CRITICAL RULES:
- "Up to £X" or "to £X" with no lower bound is NOT a salary range. comp = 0.
- Salary mentioned in the job TITLE (not jd_text) does NOT count. comp = 0.
- If no explicit £ range with two bounds appears in jd_text, comp = 0.
- Do NOT guess or infer salary. If you cannot quote the exact £ range from jd_text, comp = 0.

Converting "k" notation: £100k = £100,000. £80k-£120k = £80,000-£120,000.
- If a range is given (e.g. £80,000-£120,000), use the midpoint.

  +25 if midpoint >= £100,000
  +15 if midpoint £75,000-£99,999
  +5  if midpoint £55,000-£74,999
  -30 if midpoint < £45,000
  0   if no valid GBP salary range in jd_text

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

Example C (bad_fit — "Up to" trap):
Title: "Senior Engineer | Up to £150K" | Location: "London" | JD says "Up to £150,000" but no lower bound.
→ loc: 25 (London)
→ role: 25 (Senior)
→ tech: 0 (no matching tech)
→ comp: 0 ("Up to £150K" has no lower bound → NOT a range → comp = 0)
→ total: 25+25+0+0 = 50 → maybe
{"loc":25,"role":25,"tech":0,"comp":0,"score":50,"label":"maybe","reasoning":"Up to £150K is not a range (no lower bound), so comp=0."}

===== OUTPUT FORMAT =====

Score each category independently, then total = max(0, min(100, loc+role+tech+comp)).

Label MUST match total:
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

Return ONLY this JSON:
{"loc":0,"role":0,"tech":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"brief"}

```
