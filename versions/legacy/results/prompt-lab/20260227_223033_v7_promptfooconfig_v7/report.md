# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v7 |
| **Config** | promptfooconfig_v7.yaml |
| **Timestamp** | 2026-02-27T22:36:00.290Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 40% |
| Parse Fail | 0% |
| Score MAE | 25 |
| Score Bias | +10 |
| Avg Latency | 32.4s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 3 | 1 | 0 | 0 |
| **maybe** | 1 | 1 | 1 | 0 |
| **bad_fit** | 1 | 2 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (45) | 70/45/-25 | 26.9s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [MISS] | maybe (50) | bad_fit (0) | 50/0/-50 | 27.6s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (75) | 65/75/+10 | 29.9s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (75) | 75/75/+0 | 32.7s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 32.9s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [OK] | maybe (50) | maybe (50) | 50/50/+0 | 28.9s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (50) | 0/50/+50 | 41.5s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (70) | 25/70/+45 | 35.8s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (90) | 80/90/+10 | 32.2s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [MISS] | bad_fit (0) | maybe (60) | 0/60/+60 | 35.4s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (6/10)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 45)
- **Score error:** -25
- **Latency:** 26.9s

> **Model reasoning:** Manchester (+10), no seniority keyword (0), AI/ML experience required (+10), midpoint £105,000 (+25) = 45.

### Miss 2: 4371308941 | Senior Backend Engineer @ ScaleXP

- **Expected:** maybe (score 50)
- **Predicted:** bad_fit (score 0)
- **Score error:** -50
- **Latency:** 27.6s

> **Model reasoning:** Hybrid London (+25), no Senior/Staff/Lead keywords (0), no tech stack identified (0), no GBP salary stated (0).

### Miss 3: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 75)
- **Score error:** +10
- **Latency:** 29.9s

> **Model reasoning:** Hybrid London (+25), Senior (+25), AI/ML experience required (+25) = 75

### Miss 4: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 50)
- **Score error:** +50
- **Latency:** 41.5s

> **Model reasoning:** Location: United States (+50), Role: No Seniority Keywords (0), Tech: React (0), Compensation: No GBP Salary (0). Total: 50.

### Miss 5: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 70)
- **Score error:** +45
- **Latency:** 35.8s

> **Model reasoning:** Hybrid London (+25), Senior keywords present (+15), Node.js (+10), TypeScript (+5), Midpoint salary £170,000 >= £100,000 (+25) = 70

### Miss 6: 4375738047 | Full Stack Engineer @ Quikr

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 60)
- **Score error:** +60
- **Latency:** 35.4s

> **Model reasoning:** Remote (United States) (+25), Full Stack (+15), Node.js (+10), TypeScript (+5), no GBP salary (0) = 60.

## Prompt Used

```
You are a mechanical job-fit scorer. Follow every rule exactly. Score ONLY what is explicitly written. Do NOT infer or assume.

Inputs:
- Job title: {{job_title}}
- Job location: {{job_location}}
- Job description: {{jd_text}}

===== STEP 1: LOCATION =====

Use ONLY the job_location field. Ignore any locations mentioned inside jd_text.

  -50 if job_location is outside the UK (e.g. United States, US, USA, India, Germany, California, Bangalore, Sydney, Remote (US), Seattle, Canada)
  +25 if Remote (UK/Global/Worldwide) OR hybrid/on-site in London
  +10 if hybrid/on-site in UK city that is NOT London (e.g. Manchester, Bristol, Edinburgh)
  0   if location is missing or unclear

===== STEP 2: ROLE & SENIORITY (max 25) =====

Search job_title for these EXACT keywords (case-insensitive):
  +25 keywords: Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding
  +15 keywords: Full Stack, Full-Stack, Fullstack, Mid-Level, Mid Level, Midlevel, Software Engineer II, Engineer II, SWE II
  0   if NONE of the above keywords appear in job_title

These words are worth ZERO — do NOT award points for them:
  Engineer, Developer, Backend, Frontend, Architect, Manager, Product, Systems, DevOps

Zero-point examples:
  "Backend Systems Developer" → role = 0 (no +25/+15 keyword)
  "Front End Software Developer" → role = 0 (no +25/+15 keyword)
  "DevOps Architect" → role = 0 (no +25/+15 keyword)

===== STEP 3: TECH STACK (max 25, points stack, cap at 25) =====

Search jd_text requirements/qualifications sections ONLY:
  +10 if "Node.js" or "NodeJS" is listed as required/core (NOT just "Node" alone)
  +5  if "JavaScript" or "TypeScript" is listed as required/core
  +10 if AI/ML/LLM experience is explicitly REQUIRED

Do NOT award AI points if AI is only in company description, mission, or "nice to have."
ZERO points for: React, PostgreSQL, Python, Go, Golang, Rust, Java, C#, C++, Next.js, Express.

===== STEP 4: COMPENSATION (max 25) =====

From jd_text ONLY. Count ONLY base salary explicitly in GBP (£).
Ignore: bonuses, equity, benefits, USD ($), EUR, and all non-GBP amounts.

RULES:
- "Up to £X" or "up to £Xk" with NO lower bound → comp = 0. This is NOT a salary range.
- A valid range needs BOTH lower AND upper bounds (e.g. £80,000-£120,000). Use the midpoint.
- £Xk = £X,000 (e.g. £80k = £80,000). Convert before comparing.

  +25 if midpoint >= £100,000
  +15 if midpoint £75,000-£99,999
  +5  if midpoint £55,000-£74,999
  -30 if midpoint < £45,000
  0   if no GBP salary stated or "up to" with no lower bound

===== STEP 5: CALCULATE =====

Add the four scores: total = loc + role + tech + comp.
If total < 0, set total = 0.
If total > 100, set total = 100.

Label (MUST match total):
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

===== WORKED EXAMPLES =====

Example A — good_fit:
Title: "Senior Software Engineer" | Location: "London Area, United Kingdom (Hybrid)"
JD: requires Node.js, TypeScript, salary £90,000-£110,000.
loc=25 | role=25 (Senior) | tech=15 (Node.js +10, TypeScript +5) | comp=25 (midpoint £100,000)
25+25+15+25 = 90 → good_fit
{"loc":25,"role":25,"tech":15,"comp":25,"score":90,"label":"good_fit","reasoning":"London (+25), Senior (+25), Node.js/TS (+15), midpoint £100k (+25) = 90."}

Example B — maybe:
Title: "Backend Developer" | Location: "Hybrid - Manchester, UK"
JD: requires Node.js, TypeScript, salary £100,000-£120,000.
loc=10 | role=0 (Backend Developer has no +25/+15 keyword) | tech=15 (Node.js +10, TypeScript +5) | comp=25 (midpoint £110,000)
10+0+15+25 = 50 → maybe
{"loc":10,"role":0,"tech":15,"comp":25,"score":50,"label":"maybe","reasoning":"Manchester (+10), no seniority keyword (0), Node.js/TS (+15), midpoint £110k (+25) = 50."}

Example C — bad_fit:
Title: "Full Stack Engineer" | Location: "United States"
JD: mentions React, Node.js, TypeScript, salary "competitive."
loc=-50 | role=15 (Full Stack) | tech=15 (Node.js +10, TypeScript +5; React=0) | comp=0 (no GBP)
-50+15+15+0 = -20 → floor to 0 → bad_fit
{"loc":-50,"role":15,"tech":15,"comp":0,"score":0,"label":"bad_fit","reasoning":"US location (-50), Full Stack (+15), Node.js/TS (+15), no GBP (0) = -20 floored to 0."}

===== OUTPUT =====

Return ONLY this JSON (no other text):
{"loc":0,"role":0,"tech":0,"comp":0,"score":0,"label":"","reasoning":""}

```
