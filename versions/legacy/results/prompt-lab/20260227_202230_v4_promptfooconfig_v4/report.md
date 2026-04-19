# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | v4 |
| **Config** | promptfooconfig_v4.yaml |
| **Timestamp** | 2026-02-27T20:26:03.068Z |
| **Jobs** | 10 |
| **Status** | PASS |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 60% |
| Parse Fail | 0% |
| Score MAE | 27 |
| Score Bias | +26 |
| Avg Latency | 21s |
| Tests Run | 10 |

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 3 | 1 | 0 | 0 |
| **maybe** | 1 | 2 | 0 | 0 |
| **bad_fit** | 1 | 1 | 1 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [OK] | good_fit (70) | good_fit (85) | 70/85/+15 | 15.3s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [OK] | maybe (50) | maybe (60) | 50/60/+10 | 18.6s | 4371308941 | Senior Backend Engineer @ ScaleXP |
| 3 | [OK] | maybe (65) | maybe (75) | 65/75/+10 | 26.9s | 4374147500 | Senior Backend Engineer @ Happl |
| 4 | [OK] | good_fit (75) | good_fit (90) | 75/90/+15 | 14.8s | 1785765898 | Senior Product Engineer @ Jenkins ... |
| 5 | [OK] | good_fit (90) | good_fit (90) | 90/90/+0 | 19.3s | 4364768645 | Senior Software Engineer @ Maxwell... |
| 6 | [MISS] | maybe (50) | good_fit (90) | 50/90/+40 | 18.0s | 7820757944 | Backend Systems Developer @ Spence... |
| 7 | [MISS] | bad_fit (0) | maybe (75) | 0/75/+75 | 33.7s | 4374501376 | Front End Software Develper @ Oracle |
| 8 | [MISS] | bad_fit (25) | good_fit (95) | 25/95/+70 | 21.6s | 4374122289 | Fintech & Tech for Good | Profitab... |
| 9 | [MISS] | good_fit (80) | maybe (75) | 80/75/-5 | 19.4s | 4373815463 | Senior Software Engineer @ Tact |
| 10 | [OK] | bad_fit (0) | bad_fit (30) | 0/30/+30 | 22.8s | 4375738047 | Full Stack Engineer @ Quikr |

## Misses (4/10)

### Miss 1: 7820757944 | Backend Systems Developer @ Spencer Inc

- **Expected:** maybe (score 50)
- **Predicted:** good_fit (score 90)
- **Score error:** +40
- **Latency:** 18.0s

> **Model reasoning:** This role is a strong fit due to the requirement for Node.js and TypeScript, the Senior level mentorship responsibilities, and the high salary (£120k - £140k). The hybrid location in Bristol, UK aligns with the candidate's remote preference.

### Miss 2: 4374501376 | Front End Software Develper @ Oracle

- **Expected:** bad_fit (score 0)
- **Predicted:** maybe (score 75)
- **Score error:** +75
- **Latency:** 33.7s

> **Model reasoning:** This role is a good fit due to the requirement of JavaScript/TypeScript and the focus on frontend development. However, it lacks explicit AI/ML/LLM experience, and the backend collaboration experience is only preferred, not required. The salary range is £79,800 - £178,100, which is a good fit for the target salary, and the location is not explicitly stated but appears to be UK-based given the benefits offered. The role is a Senior level, but not a Tech Lead.

### Miss 3: 4374122289 | Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN @ Owen Thomas | Pending B Corp™

- **Expected:** bad_fit (score 25)
- **Predicted:** good_fit (score 95)
- **Score error:** +70
- **Latency:** 21.6s

> **Model reasoning:** This role is a strong fit due to the Senior/Staff/Principal level, the explicit requirement of React, TypeScript, and Node.js, and the high salary (£170k+). The focus on product lifecycle and architectural decisions aligns well with a Tech Lead role, and the remote-friendly nature (potential for international offices) satisfies the candidate's location preference.

### Miss 4: 4373815463 | Senior Software Engineer @ Tact

- **Expected:** good_fit (score 80)
- **Predicted:** maybe (score 75)
- **Score error:** -5
- **Latency:** 19.4s

> **Model reasoning:** The role is a Senior Software Engineer with 8+ years experience and requires TypeScript, Node.js, and React, aligning with the candidate's preferences. However, it's hybrid in Waterloo (outside London) and the salary is £85k, which is slightly above the target, but not significantly so.

## Prompt Used

```
You are a job-fit scorer for a single candidate profile. You do your best to make sure jobs are scored correctly for the Candidate profile.

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
