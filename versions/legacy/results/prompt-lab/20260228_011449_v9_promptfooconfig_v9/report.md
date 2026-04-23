# Prompt Lab Report

| | |
|---|---|
| **Model** | wizardlm-2-7b |
| **Tag** | v9 |
| **Config** | wizardlm-2-7b/promptfooconfig_v9.yaml |
| **Timestamp** | 2026-02-28T01:22:43.588Z |
| **Jobs** | 10 |
| **Status** | FAIL* |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 60% |
| Parse Fail | 0% |
| Score MAE | 20 |
| Score Bias | +20 |
| Avg Latency | 23s |
| Tests Run | 5 |

**Fail reasons:** yes-man: all 5 predictions are "good_fit"

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 3 | 0 | 0 | 0 |
| **maybe** | 2 | 0 | 0 | 0 |
| **bad_fit** | 0 | 0 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (90) | 70/90/+20 | 22.7s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [MISS] | maybe (50) | good_fit (90) | 50/90/+40 | 23.6s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (90) | 65/90/+25 | 26.7s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (90) | 75/90/+15 | 20.3s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 21.9s | 4364768645 | Senior Software Engineer @ Maxwell... |

## Misses (2/5)

### Miss 1: 4371308941 | Senior Backend Engineer @ ScaleXP

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 90)
- **Score error:** +40
- **Latency:** 23.6s

> **Model reasoning:** Senior role in London with Node.js/TS and £100k midpoint salary.

### Miss 2: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 90)
- **Score error:** +25
- **Latency:** 26.7s

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
