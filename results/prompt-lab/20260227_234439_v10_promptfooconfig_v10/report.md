# Prompt Lab Report

| | |
|---|---|
| **Model** | qwen3-4b-instruct-2507 |
| **Tag** | v10 |
| **Config** | qwen3-4b-instruct-2507/promptfooconfig_v10.yaml |
| **Timestamp** | 2026-02-28T00:03:15.236Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 60% |
| Parse Fail | 0% |
| Score MAE | 14 |
| Score Bias | +14 |
| Avg Latency | 111.3s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 4 | 0 | 0 | 0 |
| **maybe** | 2 | 1 | 0 | 0 |
| **bad_fit** | 0 | 2 | 1 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (75) | 70/75/+5 | 120.6s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 68.6s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (65) | 65/65/+0 | 102.7s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (85) | 75/85/+10 | 230.0s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (100) | 90/100/+10 | 71.9s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (75) | 50/75/+25 | 136.4s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (30) | 0/30/+30 | 120.7s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | maybe (65) | 25/65/+40 | 48.9s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (90) | 80/90/+10 | 153.3s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 59.3s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (4/10)

### Miss 1: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 65)
- **Score error:** +0
- **Latency:** 102.7s

> **Model reasoning:** Location is London, England, United Kingdom (Hybrid) → +25. Job title is 'Senior Backend Engineer' → contains 'Senior' → +25. Tech stack: requires modern backend languages (Typescript, Golang), and mentions 'using AI tools' in workflow but not explicitly required for building AI/ML models or LLM integrations. TypeScript is listed as part of the workflow (implied in 'modern backend languages'), so +5. Node.js or JavaScript not listed. AI/ML/LLM experience is mentioned as 'using AI tools' in workflow, not as required for core development, so no +10. Thus, tech: 0. No AI/ML/LLM requirement explicitly stated (only mentioned in context of 

### Miss 2: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 75)
- **Score error:** +25
- **Latency:** 136.4s

> **Model reasoning:** Location is hybrid in Bristol, UK (10 points). Role contains 'Backend Systems Developer' — 'Backend' is not a keyword, but 'Systems Developer' does not match any senior or mid-level keywords. However, 'Developer' is not in the allowed list. The title does not contain 'Senior', 'Lead', 'Head', etc. So role should be 0. But wait: the title is 'Backend Systems Developer'. None of the exact keywords (Senior, Lead, etc.) appear. Only 'Developer' — not allowed. So role = 0. Correction: no senior keyword. So role = 0. But 'Systems Developer' — not in list. So role = 0. Re-evaluate: keywords only: Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding. None present. So role = 0. But the job says 'Backend Systems Developer' — no match. So role = 0. But wait: is 'Systems Developer' a variant? No — only exact keywords. So role = 0. But the job says 'Systems Developer' — not in list. So role = 0. But the job title is 'Backend Systems Developer' — does it contain 'Senior'? No. So role = 0. But earlier I thought it might be mid-level. But mid-level keywords: Full Stack, Mid-Level, etc. 'Systems Developer' is not one of them. So role = 0. But the job says 'Backend Systems Developer' — not matching any keyword. So role = 0. But the job description says 'you will be: architecting scalable backend systems from scratch' — implies seniority. But we must follow rules: only exact keywords. So role = 0. Tech: 'Backend' — not in list. 'Systems' — not in list. So tech = 0. Comp: 'Bristol' — location match. So comp = 10. Score = 0 + 0 + 10 = 10. But earlier I said 75. That was a mistake. Correct score: role=0, tech=0, comp=10 → total=10. So label should be 'no_fit' or 'low_fit'. But I said 'good_fit' — that was wrong. Correction: no keywords match. So role=0, tech=0, comp=10. Total=10. So score=10. Label: 'low_fit' or 'no_fit'? Threshold: 50 for good_fit. So 10 < 50 → not good_fit. So label = 'no_fit'. But I said 'good_fit' — error. Final: role=0, tech=0, comp=10, score=10, label='no_fit'.

### Miss 3: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 30)
- **Score error:** +30
- **Latency:** 120.7s

> **Model reasoning:** Location is United States → -50 (outside UK). Role contains 'Front End Software Developer' → matches 'Full-Stack' or 'Mid-Level' by inference? But only 'Full-Stack' and 'Mid-Level' are valid keywords. 'Front End' is not a valid keyword. 'Software Developer' is not a valid keyword. So no +25 or +15. However, 'Full-Stack' is not explicitly mentioned. 'Mid-Level' is not explicitly mentioned. So role score = 0. Tech stack: JavaScript/TypeScript required → +5. React is not required (only mentioned in responsibilities). Node.js not required. AI/ML not explicitly required. So tech score = 5. Compensation: salary in USD, not GBP → 0. Final score = 0 + 0 + 5 + 0 = 5. But wait — recheck: job title is 'Front End Software Developer'. Does it contain 'Full Stack'? No. 'Mid-Level'? No. 'Senior'? No. So role = 0. Tech: JavaScript/TypeScript → +5. AI/ML/LLM? Not explicitly required. So 0. Location: United States → -50. Compensation: salary in USD, not GBP → 0. Final score: -50. ✅ All rules followed. No inference. Only explicit matches. Final score: **-50**. ✅ Done. (Note: The job title is 

### Miss 4: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** maybe (score 65)
- **Score error:** +40
- **Latency:** 48.9s

> **Model reasoning:** Location is London Area (UK) hybrid → +25. Role contains 'Senior' → +25. Tech stack includes Node.js and TypeScript → +10 (Node.js) +5 (TypeScript) = +15. Salary is 'Up to £150K' with no lower bound → comp = 0.

## Prompt Used

```
You are a mechanical job-fit scorer. Follow every rule exactly. Score ONLY what is explicitly written. Do NOT infer or assume.

Inputs:
- Job title: {{job_title}}
- Job location: {{job_location}}
- Job description: {{jd_text}}

If job_title or job_location is empty, extract them from the job description.

===== STEP 1: LOCATION (score this FIRST) =====

Use the job_location field (or extracted location if the field was empty).

  -50 if location is outside the UK (e.g. United States, US, USA, India, Germany, California, Bangalore, Sydney, Remote (US), Seattle, Canada)
  +25 if Remote (UK/Global/Worldwide) OR hybrid/on-site in London
  +10 if hybrid/on-site in UK city that is NOT London (e.g. Manchester, Bristol, Edinburgh)
  0   if location is missing or unclear

===== STEP 2: ROLE & SENIORITY (max 25) =====

Search job_title (or extracted title) for these EXACT keywords (case-insensitive):
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

CAUTION: Read these rules carefully before scoring.

From jd_text ONLY. Count ONLY base salary explicitly in GBP (£).
Ignore: bonuses, equity, benefits, USD ($), EUR, and all non-GBP amounts.

Converting "k" notation: £100k = £100,000. £80k-£120k = £80,000-£120,000.

FIRST check: Is the salary stated as "Up to £X" or "to £X" with NO lower bound?
  → If YES: comp = 0. STOP. Do not score further.

If a proper range is given (e.g. £80,000-£120,000), use the midpoint:
  +25 if midpoint >= £100,000
  +15 if midpoint £75,000-£99,999
  +5  if midpoint £55,000-£74,999
  -30 if midpoint < £45,000

  0   if no GBP salary stated

===== WORKED EXAMPLES =====

Example A (good_fit):
Title: "Senior Software Engineer" | Location: "London Area, United Kingdom (Hybrid)" | JD requires Node.js, TypeScript, salary £90,000-£110,000.
→ loc: 25 (London, UK)
→ role: 25 (Senior)
→ tech: 15 (Node.js +10, TypeScript +5)
→ comp: 25 (range £90k-£110k, midpoint £100,000 → +25)
→ total: 25+25+15+25 = 90 → good_fit
{"loc":25,"role":25,"tech":15,"comp":25,"score":90,"label":"good_fit","reasoning":"Senior role in London with Node.js/TS and £100k midpoint salary."}

Example B (bad_fit — outside UK):
Title: "Full Stack Engineer" | Location: "United States" | JD mentions React, Node.js, TypeScript, competitive salary.
→ loc: -50 (United States = outside UK)
→ role: 15 (Full Stack)
→ tech: 15 (Node.js +10, TypeScript +5; React = 0)
→ comp: 0 (no GBP salary)
→ total: max(0, -50+15+15+0) = 0 → bad_fit
{"loc":-50,"role":15,"tech":15,"comp":0,"score":0,"label":"bad_fit","reasoning":"US location (-50) makes total negative, floored to 0."}

Example C (maybe — "Up to" salary):
Title: "Senior Engineer" | Location: "London" | JD requires TypeScript, salary "Up to £150K".
→ loc: 25 (London)
→ role: 25 (Senior)
→ tech: 5 (TypeScript +5, no Node.js)
→ comp: 0 ("Up to £150K" has no lower bound → comp = 0)
→ total: 25+25+5+0 = 55 → maybe
{"loc":25,"role":25,"tech":5,"comp":0,"score":55,"label":"maybe","reasoning":"Senior in London, TS only, 'Up to' salary = no range so comp 0."}

===== OUTPUT FORMAT =====

Compute each category, then total = max(0, min(100, loc+role+tech+comp)).

Label from total:
  good_fit: 70-100
  maybe:    50-69
  bad_fit:  0-49

Your JSON values MUST match your final computed numbers. Return ONLY this JSON:
{"loc":0,"role":0,"tech":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"brief"}

```
