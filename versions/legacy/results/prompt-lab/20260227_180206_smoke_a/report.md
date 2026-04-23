# Prompt Lab Report

| | |
|---|---|
| **Model** | gemma-3-4b-it |
| **Tag** | smoke_a |
| **Timestamp** | 2026-02-27T18:02:13.493Z |
| **Jobs** | 2 |
| **Status** | FAIL* |

## Summary

| Metric | Value |
|---|---|
| Label Accuracy | 0% |
| Parse Fail | 100% |
| Score MAE | 0 |
| Score Bias | +0 |
| Avg Latency | 0s |
| Tests Run | 2 |

**Fail reasons:** Consecutive errors after 2 jobs

## Confusion Matrix

| Actual \ Predicted | good_fit | maybe | bad_fit | parse_fail |
|---|---|---|---|---|
| **good_fit** | 0 | 0 | 0 | 2 |
| **maybe** | 0 | 0 | 0 | 0 |
| **bad_fit** | 0 | 0 | 0 | 0 |

## All Jobs

| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |
|---|---|---|---|---|---|---|
| 1 | [PARSE] | good_fit (70) | — (—) | 70/—/— | 0.0s | 2448773367 | Lead DevOps Architect @ Zulauf - M... |
| 2 | [PARSE] | good_fit (70) | — (—) | 70/—/— | 0.0s | 4372524932 | Node.JS Engineer @ Digital Waffle |

## Misses (2/2)

### Miss 1: 2448773367 | Lead DevOps Architect @ Zulauf - McCullough

- **Expected:** good_fit (score 70)
- **Predicted:** PARSE FAIL (score —)
- **Latency:** 0.0s

> **Raw output:** `ERROR: Failed to create context`

### Miss 2: 4372524932 | Node.JS Engineer @ Digital Waffle

- **Expected:** good_fit (score 70)
- **Predicted:** PARSE FAIL (score —)
- **Latency:** 0.0s

> **Raw output:** `ERROR: Failed to create context`

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
