# Student Model Evaluation Report
> Generated: 2026-03-06

## Summary
| Metric | Value |
|--------|-------|
| Total jobs | 141 |
| Valid outputs | 141 |
| Parse failures | 0 |
| **Label accuracy** | **108/141 = 76.6%** |

## Field Accuracy
| Field | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| loc | 127 | 141 | 90.1% |
| role | 127 | 141 | 90.1% |
| tech | 81 | 141 | 57.4% |
| comp | 115 | 141 | 81.6% |

## Per-Label Breakdown
| Label | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| good_fit | 4 | 7 | 57% |
| maybe | 34 | 53 | 64% |
| bad_fit | 70 | 81 | 86% |

## Confusion Matrix
| Golden \ Predicted | good_fit | maybe | bad_fit |
|---|---|---|---|
| good_fit | **4** | 2 | 1 |
| maybe | 8 | **34** | 11 |
| bad_fit | 2 | 9 | **70** |

## Field Error Transitions
### loc
| Transition | Count |
|------------|-------|
| 25→10 | 9 |
| -50→25 | 2 |
| -50→10 | 2 |
| 10→25 | 1 |

### role
| Transition | Count |
|------------|-------|
| 25→15 | 5 |
| 25→0 | 4 |
| 0→15 | 2 |
| 15→0 | 2 |
| 0→25 | 1 |

### tech
| Transition | Count |
|------------|-------|
| 15→5 | 20 |
| 10→0 | 11 |
| 25→15 | 6 |
| 0→5 | 5 |
| 0→10 | 3 |
| 0→15 | 3 |
| 15→0 | 3 |
| 10→15 | 2 |
| 20→0 | 2 |
| 25→5 | 2 |
| 20→15 | 1 |
| 5→15 | 1 |
| 15→10 | 1 |

### comp
| Transition | Count |
|------------|-------|
| 15→25 | 7 |
| 0→25 | 7 |
| 15→5 | 2 |
| 5→15 | 2 |
| 25→15 | 1 |
| 5→0 | 1 |
| 0→5 | 1 |
| 25→0 | 1 |
| 0→15 | 1 |
| 15→0 | 1 |
| 25→5 | 1 |
| 15→65 | 1 |

## Top 5 Worst Errors
### 1. Senior Full-Stack Engineer (#51)
- **Company:** DigniFi
- **Location:** United States
- **Golden:** bad_fit (score=-10)
- **Predicted:** good_fit (score=80)
- **Field diffs:** loc: -50→25, tech: 15→5, comp: 0→25
- **Reasoning:** London/Remote UK (+25), senior-level role (+25), tech stack (5), salary ≥£100k (+25).

### 2. VP of Engineering (#1)
- **Company:** Homenick - Bashirian
- **Location:** Prague, Czech Republic
- **Golden:** bad_fit (score=10)
- **Predicted:** good_fit (score=75)
- **Field diffs:** loc: -50→25, tech: 10→0
- **Reasoning:** London/Remote UK (+25), senior-level role (+25), no matching tech (0), salary ≥£100k (+25).

### 3. Senior Node React Engineer (#52)
- **Company:** Raval West
- **Location:** Sacramento, CA
- **Golden:** bad_fit (score=0)
- **Predicted:** maybe (score=50)
- **Field diffs:** loc: -50→10, tech: 25→15
- **Reasoning:** US location (+10), senior-level role (+25), tech stack (15), no GBP salary (0).

### 4. Senior Software Engineer, Product Engineering, Canada (#74)
- **Company:** Ashby
- **Location:** Waterloo, Ontario, Canada
- **Golden:** bad_fit (score=0)
- **Predicted:** maybe (score=50)
- **Field diffs:** loc: -50→10
- **Reasoning:** UK outside London (+10), senior-level role (+25), tech stack (15), no GBP salary (0).

### 5. Lead Full Stack Engineer (#98)
- **Company:** Spruce
- **Location:** Bristol, England, United Kingdom
- **Golden:** good_fit (score=75)
- **Predicted:** bad_fit (score=40)
- **Field diffs:** tech: 15→5, comp: 25→0
- **Reasoning:** UK outside London (+10), senior-level role (+25), tech stack (5), no GBP salary (0).

## Training Suggestions
1. 🔴 **[HIGH] Field error: tech 15→5** — Student under-scoring tech (15→5). Add examples where tech=15 to teach correct scoring. (20 cases)
   - Examples: Senior Full-Stack Engineer, Angular Developer, Associate Software Engineer - (graduate)
2. 🔴 **[HIGH] Label confusion: maybe→bad_fit** — Student is under-scoring maybe jobs as bad_fit (21% error rate). Add more maybe training examples with clear scoring. (11 cases)
   - Examples: Software Engineer III - AI - Chase UK, Mid Level Backend Engineer (Node.js), Full Stack Developer - Fixed Term Contract
3. 🟡 **[MEDIUM] Systematic under-scoring bias** — Student under-scores 23 more jobs than it over-scores (29 over vs 52 under). Rebalance training data toward higher-scoring examples. (23 cases)
4. 🟡 **[MEDIUM] Field error: tech 10→0** — Student missing tech=10, predicting 0. Add more examples where tech=10 with explicit evidence in the JD. (11 cases)
   - Examples: VP of Engineering, Principal Backend Engineer, Senior Lead - Staff Software Engineer Senior Lead - Staff Software Engineer with verification
5. 🟡 **[MEDIUM] Label confusion: bad_fit→maybe** — Student is over-scoring bad_fit jobs as maybe (11% error rate). Add more bad_fit training examples with clear scoring. (9 cases)
   - Examples: Fullstack Software Engineer (Python) Fullstack Software Engineer (Python) with verification, Site Reliability Engineer - Trading - £70,000-£120,000 + Bonus Site Reliability Engineer - Trading - £70,000-£120,000 + Bonus with verification, Senior Node React Engineer
6. 🟡 **[MEDIUM] Field error: loc 25→10** — Student under-scoring loc (25→10). Add examples where loc=25 to teach correct scoring. (9 cases)
   - Examples: Mid Level Backend Engineer (Node.js), Linux Systems Engineer, Software Engineer, Integrations
7. 🟡 **[MEDIUM] Label confusion: maybe→good_fit** — Student is over-scoring maybe jobs as good_fit (15% error rate). Add more maybe training examples with clear scoring. (8 cases)
   - Examples: Senior Systems Architect, Senior Data Engineer Senior Data Engineer with verification, Senior Power Platform Developer Senior Power Platform Developer with verification
8. 🟢 **[LOW] Field error: comp 15→25** — Student over-scoring comp (15→25). Add training examples where comp=15 with similar JD patterns. (7 cases)
   - Examples: Senior Systems Architect, Software Engineer | React & Python (full stack), Lead Full-Stack Developer
9. 🟢 **[LOW] Field error: comp 0→25** — Student hallucinating comp=25 when golden is 0. Add more comp=0 examples to teach the model to score conservatively. (7 cases)
   - Examples: Senior Full-Stack Engineer, System Engineer | $80/hr Remote, AI Engineer AI Engineer with verification
10. 🟢 **[LOW] Field error: tech 25→15** — Student under-scoring tech (25→15). Add examples where tech=25 to teach correct scoring. (6 cases)
   - Examples: Lead Full-Stack Developer, Fintech & Tech for Good | Profitable & Hyper-Growth Start-Up | Senior/Staff/Principal | Up to £150K | 3 Days Per Week LDN, Senior Node React Engineer
11. 🟢 **[LOW] Field error: role 25→15** — Student under-scoring role (25→15). Add examples where role=25 to teach correct scoring. (5 cases)
   - Examples: Snr Full Stack Engineer | Hyper-Growth Fintech, Director Of Engineering, Engineering Supervisor - Dayshift
12. 🟢 **[LOW] Field error: tech 0→5** — Student hallucinating tech=5 when golden is 0. Add more tech=0 examples to teach the model to score conservatively. (5 cases)
   - Examples: Software Engineer | React & Python (full stack), Senior Software Engineer - React Native/React Senior Software Engineer - React Native/React, Software Engineer - AI Agents | Rust | Typescript | React Software Engineer - AI Agents | Rust | Typescript | React
13. 🟢 **[LOW] Field error: role 25→0** — Student missing role=25, predicting 0. Add more examples where role=25 with explicit evidence in the JD. (4 cases)
   - Examples: Software Engineer III - AI - Chase UK, Software Engineer III – GenAI, Python, AWS, Software Engineer III- Data Engineer, Java/Python
14. 🟢 **[LOW] Field error: tech 0→10** — Student hallucinating tech=10 when golden is 0. Add more tech=0 examples to teach the model to score conservatively. (3 cases)
   - Examples: Senior Software Engineer (c++), Fullstack Software Engineer (Python) Fullstack Software Engineer (Python) with verification, Senior Lead Software Engineer - Python & AWS
15. 🟢 **[LOW] Field error: tech 0→15** — Student hallucinating tech=15 when golden is 0. Add more tech=0 examples to teach the model to score conservatively. (3 cases)
   - Examples: Senior Product Engineer - Python / Django / FastAPI focused, (senior) Software Engineer, Senior Software Engineer, Full Stack - Fintech Airlines - Europe (100% Remote - Uk)
16. 🟢 **[LOW] Field error: tech 15→0** — Student missing tech=15, predicting 0. Add more examples where tech=15 with explicit evidence in the JD. (3 cases)
   - Examples: Intern - Frontend Developer, Machine Learning Engineer, AI Foundations, Staff Software Engineer (team Lead) - Engine By Starling
