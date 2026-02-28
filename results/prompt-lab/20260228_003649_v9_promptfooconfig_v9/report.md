# Prompt Lab Report

| | |
|---|---|
| **Model** | meta-llama-3.1-8b-instruct |
| **Tag** | v9 |
| **Config** | meta-llama-3.1-8b-instruct/promptfooconfig_v9.yaml |
| **Timestamp** | 2026-02-28T00:40:45.846Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 60% |
| Parse Fail | 0% |
| Score MAE | 15.5 |
| Score Bias | -15.5 |
| Avg Latency | 23.1s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 1 | 3 | 0 | 0 |
| **maybe** | 0 | 2 | 1 | 0 |
| **bad_fit** | 0 | 0 | 3 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (35) | 70/35/-35 | 20.8s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [MISS] | maybe (50) | bad_fit (0) | 50/0/-50 | 25.7s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (50) | 65/50/-15 | 23.1s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [MISS] | good_fit (75) | maybe (65) | 75/65/-10 | 19.9s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 19.5s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [OK] | maybe (50) | maybe (35) | 50/35/-15 | 21.9s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 28.4s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [OK] | bad_fit (25) | bad_fit (0) | 25/0/-25 | 25.2s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [MISS] | good_fit (80) | maybe (75) | 80/75/-5 | 20.2s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 26.5s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (4/10)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 35)
- **Score error:** -35
- **Latency:** 20.8s

> **Model reasoning:** Hybrid location in Manchester, UK, but no seniority keywords in job title, no tech stack points, and £100,000-£115,000 midpoint salary.

### Miss 2: 4371308941 | Senior Backend Engineer @ ScaleXP

- **Expected:** maybe (score 50)
- **Predicted:** bad_fit (score 0)
- **Score error:** -50
- **Latency:** 25.7s

> **Model reasoning:** London Area, United Kingdom (Hybrid) is a hybrid/on-site in London location, but job title does not contain any of the required keywords for role and seniority, and job description does not mention any of the required tech stack or compensation.

### Miss 3: 1785765898 | Senior Product Engineer @ Jenkins - Zboncak

- **Expected:** good_fit (score 75)
- **Predicted:** maybe (score 65)
- **Score error:** -10
- **Latency:** 19.9s

> **Model reasoning:** Remote (UK) location, Senior Product Engineer role, but no tech stack points awarded, and compensation midpoint is £95,000.

### Miss 4: 4373815463 | Senior Software Engineer @ Tact

- **Expected:** good_fit (score 80)
- **Predicted:** maybe (score 75)
- **Score error:** -5
- **Latency:** 20.2s

> **Model reasoning:** Hybrid London role with Senior title and £85,000 salary, but no explicit tech stack mentioned.

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
