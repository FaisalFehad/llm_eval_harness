# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v6 |
| **Config** | promptfooconfig_v6.yaml |
| **Timestamp** | 2026-02-27T22:08:14.011Z |
| **Jobs** | 10 |
| **Status** | FAIL |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 50% |
| Parse Fail | 0% |
| Score MAE | 32.5 |
| Score Bias | +31.5 |
| Avg Latency | 28.8s |
| Tests Run | 10 |

**Fail reasons:** score_mae=32.5 > 30

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 4 | 0 | 0 | 0 |
| **maybe** | 2 | 1 | 0 | 0 |
| **bad_fit** | 2 | 1 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (85) | 70/85/+15 | 27.5s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (65) | 50/65/+15 | 31.8s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (85) | 65/85/+20 | 27.0s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (90) | 75/90/+15 | 30.2s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (85) | 90/85/-5 | 24.4s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (85) | 50/85/+35 | 23.2s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | good_fit (85) | 0/85/+85 | 42.3s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (90) | 25/90/+65 | 27.5s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (85) | 80/85/+5 | 23.4s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [MISS] | bad_fit (0) | maybe (65) | 0/65/+65 | 31.2s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (5/10)

### Miss 1: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 85)
- **Score error:** +20
- **Latency:** 27.0s

> **Model reasoning:** The job is for a Senior Backend Engineer in London, UK, with a hybrid working arrangement. The job description highlights experience with TypeScript, Golang, and API design, aligning with the tech stack criteria. The salary is explicitly stated as £100,000, exceeding the threshold for a good fit.

### Miss 2: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 85)
- **Score error:** +35
- **Latency:** 23.2s

> **Model reasoning:** The role is a Backend Systems Developer with a hybrid location in Bristol, UK. The job description requires Node.js, TypeScript, and PostgreSQL. The salary range of £120,000 - £140,000 falls within the good fit criteria.

### Miss 3: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** good_fit (score 85)
- **Score error:** +85
- **Latency:** 42.3s

> **Model reasoning:** The job is located in the United States, a non-UK location, resulting in a -50 score. The role is a Front End Software Developer, which earns +15 points. The tech stack includes JavaScript/TypeScript (ES6+), React, Redux, and testing frameworks, scoring +25. The compensation range is between $79,800 and $178,100 USD, with a midpoint of approximately $124,000, earning +15 points.  The total score is 85, indicating a good fit.

### Miss 4: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 90)
- **Score error:** +65
- **Latency:** 27.5s

> **Model reasoning:** The job description clearly indicates a Senior/Staff/Principal level role with a salary of up to £170K, which exceeds the threshold for a good fit. The tech stack heavily features React, TypeScript, and Node.js, aligning with the requirements. The location is London, UK, and the role is hybrid.

### Miss 5: 4375738047 | Full Stack Engineer @ Quikr

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 65)
- **Score error:** +65
- **Latency:** 31.2s

> **Model reasoning:** The job is located in the United States, resulting in a -50 score. The role is a Full Stack Engineer, earning a +15 point. The tech stack includes Node.js and TypeScript, contributing +15 points. The compensation is described as 'competitive' without a specific GBP figure, resulting in a 0 score. Overall, the fit is leaning towards maybe due to the location and compensation ambiguity.

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
Step 1 — Location: "United States" is outside UK → -50
Step 2 — Role: "Full Stack" → +15
Step 3 — Tech: Node.js +10, TypeScript +5 → +15
Step 4 — Comp: "competitive salary" no GBP figure → 0
Total: -50 + 15 + 15 + 0 = -20 → floor to 0 → bad_fit

===== CALCULATE & OUTPUT =====

total = location + role + tech + comp (floor 0, cap 100)

Label (MUST match total):
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

Return ONLY this JSON:
{"role":0,"tech":0,"loc":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"1-2 sentences"}

```
