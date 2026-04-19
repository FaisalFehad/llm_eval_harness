# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v5 |
| **Config** | promptfooconfig_v5.yaml |
| **Timestamp** | 2026-02-27T21:20:04.905Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 50% |
| Parse Fail | 0% |
| Score MAE | 25 |
| Score Bias | +22 |
| Avg Latency | 27.9s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 2 | 2 | 0 | 0 |
| **maybe** | 1 | 2 | 0 | 0 |
| **bad_fit** | 1 | 1 | 1 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [MISS] | good_fit (70) | maybe (65) | 70/65/-5 | 28.4s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (65) | 50/65/+15 | 26.7s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (90) | 65/90/+25 | 29.4s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [MISS] | good_fit (75) | maybe (65) | 75/65/-10 | 33.1s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 24.4s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [OK] | maybe (50) | maybe (65) | 50/65/+15 | 25.6s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (75) | 0/75/+75 | 38.9s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (95) | 25/95/+70 | 24.0s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (85) | 80/85/+5 | 23.7s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (30) | 0/30/+30 | 24.4s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (5/10)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** maybe (score 65)
- **Score error:** -5
- **Latency:** 28.4s

> **Model reasoning:** The role is for a Lead DevOps Architect, which scores 15 points for seniority. It requires Python and PyTorch, scoring 10 points for AI/ML. The location is hybrid in Manchester, UK, scoring 10 points. The salary range is £100,000 - £115,000, scoring 15 points. However, it doesn't explicitly require Node.js or React, and the desired remote location in London is not met.

### Miss 2: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 90)
- **Score error:** +25
- **Latency:** 29.4s

> **Model reasoning:** The job description explicitly lists 'Senior Backend Engineer' and requires experience with 'Typescript, Golang' (Node.js), and mentions 'AI tools' which aligns with the candidate's interests. The role is based in London with a hybrid work arrangement, and the salary expectation of £75k+ is met.

### Miss 3: 1785765898 | Senior Product Engineer @ Jenkins - Zboncak

- **Expected:** good_fit (score 75)
- **Predicted:** maybe (score 65)
- **Score error:** -10
- **Latency:** 33.1s

> **Model reasoning:** The role is for a Senior Product Engineer, which scores 15 points in Role & Seniority. It requires TypeScript (5 points) and Node.js (10 points) in the tech stack. The location is Remote (UK), scoring 25 points in Location & Work Arrangement. The salary range is £120,000 - £140,000, resulting in a midpoint of £130,000, scoring 25 points in Compensation. Overall, it's a potentially good fit, but the lack of AI/ML experience prevents it from being a strong one.

### Miss 4: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 75)
- **Score error:** +75
- **Latency:** 38.9s

> **Model reasoning:** The job description lists React and TypeScript as required skills, and mentions AI/ML experience. It also offers a salary range of $79,800 - $178,100 (approximately £65k - £130k), which is within the candidate's target range. However, the location is primarily focused on the US, and the work arrangement is not explicitly stated as remote or hybrid in London, limiting the score.

### Miss 5: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 95)
- **Score error:** +70
- **Latency:** 24.0s

> **Model reasoning:** The job description explicitly states 'Senior/Staff/Principal' roles, requiring strong engineering skills. It lists Node.js, React, and TypeScript as core requirements, and mentions AI/ML experience. Furthermore, the salary is up to £170K and the location is hybrid in London, aligning well with the candidate's preferences and experience.

## Prompt Used

```
You are a strict job‑fit scorer for one candidate. Score ONLY what the job description explicitly states. Do NOT infer missing info.

Candidate context (for scoring only):
- UK‑based senior/staff engineer interested in Node/React/TypeScript and AI/ML roles.
- Prefers remote (UK/global) or hybrid in London.
- Salary target £75k+.

Scoring rubric — sum four categories (floor 0, cap 100):

1) Role & Seniority (max 25)
+25: Title explicitly includes Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, or Founding
+15: Title explicitly includes Full Stack, Mid‑Level/Mid Level/Midlevel, Software Engineer II / SWE II / Engineer II
0: Otherwise

2) Tech Stack (max 25, points stack; cap at 25)
+10: Node.js OR React is explicitly required or listed in core stack/requirements
+5: JavaScript or TypeScript is explicitly required
+10: AI/ML/LLM experience is explicitly required (not just company uses AI)

3) Location & Work Arrangement (max 25)
+25: Fully remote (UK/global) OR remote (UK/England/Scotland/etc) OR hybrid/on‑site in London
+10: Hybrid/on‑site in UK outside London
0: Location not stated or unclear
-50: Location explicitly outside the UK

4) Compensation (max 25)
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

Job description:
{{jd_text}}


Candidate profile:
- UK-based; 6+ years in backend or full stack Software engineer.
- Strong in TypeScript, Node.js, React
- Interested in AI/ML/LLM roles
- Target level: Senior, Tech Lead
- Needs remote (UK or global) or hybrid in London; salary target £75k+, the higher the better

You must strictly score jobs according to the following criteria
Score this job 0–100 by summing the four categories below (floor 0, cap 100).

1. Role & Seniority (max 25 pts):
   +25: Senior, Staff, Principal, Lead, or Tech Lead Engineer
   +15: Full Stack, Mid-Level, or Software Engineer II
   0: Junior, management-only (EM), or non-engineering roles

2. Tech Stack & Domain (max 25 pts, points stack):
   +10: Node.js explicitly required
   +5: JavaScript or TypeScript explicitly required
   +10: AI, ML, or LLM experience explicitly mentioned

3. Location & Work Arrangement (max 25 pts):
   +25: Fully remote (UK or global) OR hybrid in London
   +10: Hybrid or on-site in UK outside London
   0: Location unknown
   -50: Hybrid or on-site outside the UK

4. Compensation (max 25 pts):
   +25: Base salary £100k+
   +15: Base salary £75k–£99k
   +5: Base salary £55k–£74k
   0: Salary unknown or not listed
   -30: Base salary below £45k

Label mapping:
- good_fit (70–100)
- maybe (50–69)
- bad_fit (0–49)

Output format:
Return ONLY JSON with this schema:
{"label":"good_fit|maybe|bad_fit","score":0-100,"reasoning":"1-2 sentences"}

Job description:
{{jd_text}}

```
