# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v6b |
| **Config** | promptfooconfig_v6.yaml |
| **Timestamp** | 2026-02-27T22:13:38.073Z |
| **Jobs** | 10 |
| **Status** | FAIL* |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 0% |
| Parse Fail | 0% |
| Score MAE | 70 |
| Score Bias | -70 |
| Avg Latency | 24.4s |
| Tests Run | 5 |

**Fail reasons:** yes-man: all 5 predictions are "bad_fit"

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 0 | 0 | 3 | 0 |
| **maybe** | 0 | 0 | 2 | 0 |
| **bad_fit** | 0 | 0 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | bad_fit (0) | 70/0/-70 | 22.7s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [MISS] | maybe (50) | bad_fit (0) | 50/0/-50 | 24.3s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | bad_fit (0) | 65/0/-65 | 25.3s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [MISS] | good_fit (75) | bad_fit (0) | 75/0/-75 | 22.6s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [MISS] | good_fit (90) | bad_fit (0) | 90/0/-90 | 27.0s | 4364768645 | Senior Software Engineer @ Maxwell... |

## Misses (5/5)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** bad_fit (score 0)
- **Score error:** -70
- **Latency:** 22.7s

> **Model reasoning:** Job location is Hybrid - Manchester, UK. However, the job description explicitly states 'About Zulauf - McCullough' and the location is outside the UK. Therefore, -50 points.

### Miss 2: 4371308941 | Senior Backend Engineer @ ScaleXP

- **Expected:** maybe (score 50)
- **Predicted:** bad_fit (score 0)
- **Score error:** -50
- **Latency:** 24.3s

> **Model reasoning:** Job location is London Area, United Kingdom (Hybrid), but the job description contains 'United States' which is outside the UK. This results in a -50 score.

### Miss 3: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** bad_fit (score 0)
- **Score error:** -65
- **Latency:** 25.3s

> **Model reasoning:** The job location is London, England, United Kingdom (Hybrid), but the job description explicitly states 'Remote (US)' which is outside the UK. This results in a location score of -50.

### Miss 4: 1785765898 | Senior Product Engineer @ Jenkins - Zboncak

- **Expected:** good_fit (score 75)
- **Predicted:** bad_fit (score 0)
- **Score error:** -75
- **Latency:** 22.6s

> **Model reasoning:** Job location is Remote (UK), but the job description explicitly states 'United States' which is outside the UK, resulting in a -50 score. No relevant technical skills were identified. Salary is not stated in GBP.

### Miss 5: 4364768645 | Senior Software Engineer @ Maxwell Bond

- **Expected:** good_fit (score 90)
- **Predicted:** bad_fit (score 0)
- **Score error:** -90
- **Latency:** 27.0s

> **Model reasoning:** Job location is London Area, United Kingdom (Hybrid), but the job description explicitly states 'About the job Senior Software EngineerLocation: London - Hybrid'. The location is outside the UK due to the explicit mention of 'United Kingdom' in the location description. Therefore, the location score is -50.

## Prompt Used

```
You are a mechanical job-fit scorer. Follow every rule exactly. Score ONLY what is explicitly written. Do NOT infer or assume.

Inputs:
- Job title: {{job_title}}
- Job location: {{job_location}}
- Job description: {{jd_text}}

===== STEP 1: LOCATION (score this FIRST) =====

Check job_location AND jd_text for non-UK signals.
If EITHER field names a country, state, or city OUTSIDE the United Kingdom, location = -50.
Non-UK examples: United States, US, USA, India, Germany, California, New York, Remote (US), Berlin, Bangalore, Sydney, Amsterdam, Seattle, Tel Aviv, Singapore, Canada, Dubai.

Otherwise use job_location:
  +25 if Remote (UK/Global/Worldwide) OR hybrid/on-site in London
  +10 if hybrid/on-site in UK city that is NOT London (e.g. Manchester, Bristol, Edinburgh, Birmingham)
  0   if location is missing or unclear
  -50 if location is explicitly outside the UK

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

===== WORKED EXAMPLE =====

Title: "Full Stack Engineer" | Location: "Remote (United States)" | JD mentions React, Node.js, TypeScript, competitive salary.
→ loc: -50 (United States = outside UK)
→ role: 15 (Full Stack)
→ tech: 15 (Node.js +10, TypeScript +5; React = 0)
→ comp: 0 (no GBP salary)
→ score: max(0, -50+15+15+0) = 0 → bad_fit
{"loc":-50,"role":15,"tech":15,"comp":0,"score":0,"label":"bad_fit","reasoning":"US location (-50) makes total negative, floored to 0."}

===== OUTPUT FORMAT =====

Output each category score, then total = max(0, min(100, loc+role+tech+comp)).

Label MUST match total:
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

Return ONLY this JSON (loc FIRST):
{"loc":-50,"role":0,"tech":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"brief"}

```
