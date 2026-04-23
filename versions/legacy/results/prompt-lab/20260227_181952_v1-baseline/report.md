# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v1-baseline |
| **Config** | promptfooconfig.yaml |
| **Timestamp** | 2026-02-27T18:23:24.414Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 60% |
| Parse Fail | 0% |
| Score MAE | 26.8 |
| Score Bias | +26.8 |
| Avg Latency | 20.9s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 4 | 0 | 0 | 0 |
| **maybe** | 2 | 1 | 0 | 0 |
| **bad_fit** | 1 | 1 | 1 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (85) | 70/85/+15 | 21.1s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 16.0s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [MISS] | maybe (65) | good_fit (90) | 65/90/+25 | 21.9s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (90) | 75/90/+15 | 15.0s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 17.4s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (90) | 50/90/+40 | 18.8s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (78) | 0/78/+78 | 38.0s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (90) | 25/90/+65 | 20.4s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [OK] | good_fit (80) | good_fit (90) | 80/90/+10 | 19.7s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (10) | 0/10/+10 | 20.3s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (4/10)

### Miss 1: 4374147500 | Senior Backend Engineer @ Happl

- **Expected:** maybe (score 65)
- **Predicted:** good_fit (score 90)
- **Score error:** +25
- **Latency:** 21.9s

> **Model reasoning:** This role is a Senior Backend Engineer focused on building an AI-first benefits platform, aligning with the candidate's preferences for Node.js/TypeScript and AI/ML. The location is London, offering a hybrid work arrangement, and the salary range is within the target (£75k+).

### Miss 2: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 90)
- **Score error:** +40
- **Latency:** 18.8s

> **Model reasoning:** This role is a Senior Backend Systems Developer, aligning with the candidate's target seniority. It utilizes Node.js and TypeScript, and the salary is well above the target range. The location is hybrid in Bristol, UK, which is acceptable given the candidate's openness to hybrid roles.

### Miss 3: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 78)
- **Score error:** +78
- **Latency:** 38.0s

> **Model reasoning:** The role is a Senior Frontend Engineer focusing on building SPAs with React and TypeScript, aligning well with the candidate's preferred stack. The AI/ML focus is present in the description of 'AI-powered solutions.' However, the backend experience is preferred, not required, and the salary range is £79,800 - £178,100 (USD), which is slightly above the target of £75,000+. The location is London, which is a good fit, and the work arrangement is hybrid, also acceptable. The tech stack is a strong match, but the backend preference and salary range slightly reduce the score.

### Miss 4: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 90)
- **Score error:** +65
- **Latency:** 20.4s

> **Model reasoning:** This role is a Senior/Staff/Principal level, utilizes Node.js and TypeScript, and offers a high salary (£170k+). The focus on product engineering and impact aligns with the candidate's interest in AI/ML, and the location is in London, making it a strong fit.

## Prompt Used

```
You are a job-fit scoring assistant for a senior software engineer job search.

Candidate profile:
- Based in the UK; open to remote (UK or global) or hybrid roles in London/major UK cities
- Targeting: Senior, Staff, Principal, or Lead Software Engineer (backend or full-stack)
- Preferred stack: Node.js, TypeScript, JavaScript
- Interested in AI/ML/LLM roles
- Salary target: £75,000+ GBP
- Not interested in: management-only roles, non-engineering roles, or on-site roles outside the UK

Score this job 0–100 using the four categories below. Sum the points (floor 0, cap 100).

1. Role & Seniority (max 25 pts):
   +25: Senior, Staff, Principal, Lead, or Tech Lead Engineer
   +15: Full Stack, Mid-Level, or Software Engineer II
   0: Junior, management-only (EM), or non-engineering roles

2. Tech Stack & Domain (max 25 pts, points stack, cap at 25):
   +10: Node.js explicitly required
   +5: JavaScript or TypeScript explicitly required
   +10: AI, ML, or LLM experience explicitly mentioned

3. Location & Work Arrangement (max 25 pts):
   +25: Fully remote (UK-based or global) OR hybrid in London
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

Respond ONLY with a valid JSON object. No markdown, no extra text.
Format: {"label": "good_fit|maybe|bad_fit", "score": <integer 0-100>, "reasoning": "<1-2 sentences>"}

Job description:
{{jd_text}}

```
