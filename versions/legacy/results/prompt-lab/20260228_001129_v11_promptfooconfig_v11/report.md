# Prompt Lab Report

| | |
|---|---|
| **Model** | qwen3-4b-instruct-2507 |
| **Tag** | v11 |
| **Config** | qwen3-4b-instruct-2507/promptfooconfig_v11.yaml |
| **Timestamp** | 2026-02-28T00:16:06.050Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 70% |
| Parse Fail | 0% |
| Score MAE | 15 |
| Score Bias | +15 |
| Avg Latency | 27.4s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 4 | 0 | 0 | 0 |
| **maybe** | 2 | 1 | 0 | 0 |
| **bad_fit** | 1 | 0 | 2 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (75) | 70/75/+5 | 17.5s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 22.0s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (90) | 65/90/+25 | 24.4s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (85) | 75/85/+10 | 36.8s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 22.6s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (75) | 50/75/+25 | 52.8s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 28.6s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (90) | 25/90/+65 | 22.2s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (90) | 80/90/+10 | 26.5s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (0) | 0/0/+0 | 20.9s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (3/10)

### Miss 1: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 90)
- **Score error:** +25
- **Latency:** 24.4s

> **Model reasoning:** Location is London, England, UK (Hybrid) → +25. Title contains 'Senior' → +25. JD requires TypeScript and AI tools as part of workflow → TypeScript +5, AI/ML/LLM experience explicitly required (using AI tools in workflow) → +10. Salary range £80k-£120k → midpoint £100,000 → +25. Total: 25+25+15+25 = 90.

### Miss 2: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 75)
- **Score error:** +25
- **Latency:** 52.8s

> **Model reasoning:** Location is hybrid in Bristol, UK (10 points). Role contains 'Backend Systems Developer' — no explicit Senior/Lead keywords, so role score is 0. However, 'Backend Systems Developer' does not match any of the senior keywords (Senior, Lead, etc.), so role should be 0. Correction: 'Backend' is not a valid keyword; 'Systems Developer' is not a senior keyword. Only valid keywords are Senior, Staff, etc. So no senior keyword present. Therefore, role score must be 0. Re-evaluate role: title is 'Backend Systems Developer' — contains no Senior, Lead, etc. So role = 0. Tech: Node.js and TypeScript explicitly listed → +10 (Node.js) +5 (TypeScript) = 15. Compensation: £120,000-£140,000 → midpoint = £130,000 → ≥ £100,000 → +25. Location: hybrid in Bristol (UK city not London) → +10. So: loc=10, role=0, tech=15, comp=25 → total = 50 → maybe_fit. But wait: role does not contain any of the senior keywords. So role = 0. Final: loc=10, role=0, tech=15, comp=25 → sum = 45 → maybe_fit. But let's recheck role keywords: 'Backend Systems Developer' — does it contain 'Full Stack', 'Mid-Level', 'Software Engineer II'? No. Does it contain 'Senior'? No. So role = 0. Therefore: total = 10+0+15+25 = 50 → maybe_fit. But the instruction says: 'Do NOT give points for: Engineer, Developer, Backend, Frontend, Architect, Manager, Product, or any word not listed above.' So only if the title contains one of the listed senior/lead keywords. It does not. So role = 0. Final score: 50 → maybe_fit. But the output must be strictly based on rules. Let's recompute step by step.

### Miss 3: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 90)
- **Score error:** +65
- **Latency:** 22.2s

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
- "Up to £X" or "to £X" (no lower bound) is NOT a range. comp = 0.

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
