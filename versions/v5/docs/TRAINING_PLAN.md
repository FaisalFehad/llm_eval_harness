# Student Model Training Plan V5 (Final)

## Core Architecture Change: Semantic Tokens

The student model no longer outputs numeric scores. It outputs semantic labels for each field. Your code converts these to numbers and computes the total score and label deterministically.

### The Schema

```json
{
  "reasoning": "loc: 'London' in location → LONDON_OR_REMOTE. role: 'Senior' in title → SENIOR_PLUS. tech: node.js found, typescript found, react ignored → NODE_JS_TS. comp: '£85k-£120k' midpoint £102.5k → ABOVE_100K.",
  "loc": "LONDON_OR_REMOTE",
  "role": "SENIOR_PLUS",
  "tech": "NODE_JS_TS",
  "comp": "ABOVE_100K"
}
```

No `score` field. No `label` field. Your code computes both.

### Semantic Token Vocabulary

**loc (4 tokens):**
| Token | Score | Rule |
|-------|-------|------|
| LONDON_OR_REMOTE | 25 | Location contains "London" or "Remote" |
| UK_OTHER | 10 | UK but not London, not Remote |
| OUTSIDE_UK | -50 | Not in the United Kingdom |
| MISSING | 0 | No location or unclear |

**role (3 tokens):**
| Token | Score | Rule |
|-------|-------|------|
| SENIOR_PLUS | 25 | Senior, Staff, Principal, Lead, Tech Lead, Head, VP, Snr, Founding, Engineer III, SWE III |
| MID_LEVEL | 15 | Full Stack, Mid-Level, SWE II, Engineer II, or fallback senior keyword in JD |
| NO_SENIORITY | 0 | No seniority keywords found |

**tech (8 tokens):**
| Token | Score | What was found |
|-------|-------|---------------|
| NONE | 0 | No Node, no JS/TS, no AI/ML required |
| JS_TS | 5 | JavaScript or TypeScript only |
| NODE | 10 | Node.js/NodeJS only |
| NODE_JS_TS | 15 | Node + JavaScript/TypeScript |
| AI_ML | 10 | AI/ML/LLM required only |
| JS_TS_AI_ML | 15 | JS/TS + AI/ML required |
| NODE_AI_ML | 20 | Node + AI/ML required |
| NODE_JS_TS_AI_ML | 25 | Node + JS/TS + AI/ML required |

**comp (6 tokens):**
| Token | Score | Rule |
|-------|-------|------|
| NO_GBP | 0 | No £ salary found, or non-GBP currency only |
| UP_TO_ONLY | 0 | "Up to £X" or "to £X" with no lower bound |
| BELOW_45K | -30 | GBP midpoint < £45,000 |
| RANGE_55_74K | 5 | GBP midpoint £55,000–£74,999 |
| RANGE_75_99K | 15 | GBP midpoint £75,000–£99,999 |
| ABOVE_100K | 25 | GBP midpoint ≥ £100,000 |

Note: midpoint £45,000-£54,999 maps to NO_GBP (score 0). Daily rates, contract rates, and non-GBP currencies all map to NO_GBP.

### Code Layer (Python)

```python
LOC_MAP = {"LONDON_OR_REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "MISSING": 0}
ROLE_MAP = {"SENIOR_PLUS": 25, "MID_LEVEL": 15, "NO_SENIORITY": 0}
TECH_MAP = {"NONE": 0, "JS_TS": 5, "NODE": 10, "NODE_JS_TS": 15,
            "AI_ML": 10, "JS_TS_AI_ML": 15, "NODE_AI_ML": 20, "NODE_JS_TS_AI_ML": 25}
COMP_MAP = {"NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30,
            "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25}

def compute(pred):
    loc = LOC_MAP[pred["loc"]]
    role = ROLE_MAP[pred["role"]]
    tech = TECH_MAP[pred["tech"]]
    comp = COMP_MAP[pred["comp"]]
    score = max(0, min(100, loc + role + tech + comp))
    if score >= 70: label = "good_fit"
    elif score >= 50: label = "maybe"
    else: label = "bad_fit"
    return {"loc": loc, "role": role, "tech": tech, "comp": comp, "score": score, "label": label}
```

### Why This Works for 0.5B

The model now solves four independent classification problems:
- loc: 4 classes
- role: 3 classes  
- tech: 8 classes
- comp: 6 classes

Total: 21 possible output tokens across 4 fields. No arithmetic. No derived fields. Each classification is directly tied to vocabulary the model sees in the input text. "Found London" → next token probability for LONDON_OR_REMOTE is extremely high because the reasoning just primed that exact concept.

---

## The Two Prompts

### Teacher Prompt (GPT-4o-mini only)

```
You are a job-fit classifier. Classify the job below into semantic categories using ONLY these rules. Do not infer or assume. Output ONLY valid JSON.

Inputs:
- Title: {{job_title}}
- Location: {{job_location}}
- Description: {{jd_text}}

LOCATION — use ONLY the location field. Ignore the description. Ignore "City Of" / "Greater".
LONDON_OR_REMOTE: Contains "London" or "Remote" (any case/format).
UK_OTHER: Inside the UK but not London, not Remote.
OUTSIDE_UK: Not in the United Kingdom. This includes Ireland (Republic of), all EU/US/Asia/etc.
MISSING: No location provided or unclear.

ROLE — check title only (case-insensitive). Highest match wins.
SENIOR_PLUS: senior, staff, principal, lead, tech lead, head, distinguished, vp, snr, founding, engineer iii, swe iii.
MID_LEVEL: full stack, full-stack, fullstack, mid-level, midlevel, software engineer ii, engineer ii, swe ii.
Fallback: if no title match, check the first hiring sentence of the description. If it contains senior/staff/principal/lead/tech lead → MID_LEVEL. Otherwise → NO_SENIORITY.

TECH — scan the full description. Case-insensitive. Classify what combination was found.
- node.js / node / nodejs → counts as "Node"
- javascript / typescript → counts as "JS/TS" (once, even if both appear)
- AI/ML/LLM experience explicitly REQUIRED (not "nice to have", not company description) → counts as "AI/ML"
- All other tech (react, python, go, java, ruby, .net, etc.) = ignored. You MUST mention any prominent ignored tech in your reasoning with "→ ignored".
Then classify:
NONE: No Node, no JS/TS, no required AI/ML found.
JS_TS: JS/TS only.
NODE: Node only.
NODE_JS_TS: Node + JS/TS.
AI_ML: AI/ML only.
JS_TS_AI_ML: JS/TS + AI/ML.
NODE_AI_ML: Node + AI/ML.
NODE_JS_TS_AI_ML: Node + JS/TS + AI/ML.

COMP — scan description for £ followed by a number. Case-insensitive.
- Ignore all non-GBP currencies (USD $, EUR €, AUD, etc.). Mention them in reasoning with "→ ignored".
- Ignore daily rates (£X/day, £X per day, £X p/d). Mention in reasoning with "→ daily rate ignored".
- "Up to £X" or "to £X" with no lower bound → UP_TO_ONLY.
- Convert "k" to thousands. Use midpoint of ranges.
NO_GBP: No GBP annual salary found, or only non-GBP/daily rates present, or midpoint £45k-£54,999.
UP_TO_ONLY: "Up to £X" / "to £X" with no lower bound.
BELOW_45K: GBP midpoint < £45,000.
RANGE_55_74K: GBP midpoint £55,000–£74,999.
RANGE_75_99K: GBP midpoint £75,000–£99,999.
ABOVE_100K: GBP midpoint ≥ £100,000.

REASONING FORMAT — cite what you found AND what you ignored for each field. Examples:

loc: 'London, England, United Kingdom' → London found → LONDON_OR_REMOTE. role: 'Senior' in title → SENIOR_PLUS. tech: node.js found, typescript found, react ignored, python ignored → NODE_JS_TS. comp: '£85,000-£120,000' midpoint £102.5k → ABOVE_100K.

loc: 'San Francisco, CA, United States' → not UK → OUTSIDE_UK. role: 'Senior' in title → SENIOR_PLUS. tech: node.js found, javascript found, go ignored → NODE_JS_TS. comp: '$150,000-$200,000' USD ignored, no GBP → NO_GBP.

loc: 'Manchester, England' → UK not London/Remote → UK_OTHER. role: 'Full Stack' in title → MID_LEVEL. tech: typescript found, no node → JS_TS. comp: 'Up to £90,000' no lower bound → UP_TO_ONLY.

loc: 'Dublin, Ireland' → Ireland (not UK) → OUTSIDE_UK. role: 'Engineer III' in title → SENIOR_PLUS. tech: python found but ignored, java found but ignored, no node, no js/ts → NONE. comp: '€80,000-€100,000' EUR ignored → NO_GBP.

loc: 'Belfast, Northern Ireland' → UK → UK_OTHER. role: 'Software Developer' no keyword → checked JD → no senior keyword → NO_SENIORITY. tech: react found but ignored → NONE. comp: '£350/day' daily rate ignored → NO_GBP.

Return ONLY this JSON:
{"reasoning":"...","loc":"MISSING","role":"NO_SENIORITY","tech":"NONE","comp":"NO_GBP"}
```

### Student Prompt (training data + inference)

```
Classify this job. Output JSON only.

Title: {{job_title}}
Location: {{job_location}}
Description: {{jd_text}}

{"reasoning":"...","loc":"...","role":"...","tech":"...","comp":"..."}
```

The student prompt is deliberately bare. No rules, no keyword lists, no examples. The model learns classification purely from 800 training examples where the reasoning shows the thought process.

---

## Execution Plan

### Step 1: Label All Data + Fill Gaps

**1A — Label everything in one batch**

Send all ~1553 jobs through GPT-4o-mini batch API using the teacher prompt:
- 450 existing training jobs
- 640 new preprocessed jobs
- 245 salary augmented jobs
- 68 contrastive pairs
- 50 location variants
- 100 truncated JDs

**1B — Validate and normalise outputs**

- Parse all JSON. Log parse failures.
- Validate each field value is a legal semantic token. Reject if not.
- Run format normaliser on reasoning: must contain `loc:`, `role:`, `tech:`, `comp:`, `→`, and each segment must end with the corresponding token name.
- Cross-check: if reasoning says "node.js found" but tech token is NONE or JS_TS, flag as inconsistent.
- Re-label failures and inconsistencies at temperature=0.

**1C — Check distributions and generate synthetic data for gaps**

Count distributions across all labeled data. For each minimum below that isn't met, generate synthetic jobs.

Synthetic job generation prompt:
```
Generate a realistic job posting as JSON: {title, company, location, jd_text}.

Requirements:
- {{specific gap requirements}}
- jd_text: 200-400 words
- Include: equal opportunity statement, 2-3 sentence company description, benefits section
- Use inconsistent formatting: mix bullet points with prose, vary capitalisation of tech names
- Include at least one misleading detail: e.g. mention a different city in company description, include a non-GBP salary alongside GBP, list a scored tech as "nice to have" while listing unscored tech as "required"
```

Post-process all synthetic JDs: randomly change tech name capitalisation (node.js → NodeJS → nodejs), remove some bullet formatting, add realistic typos in non-critical words.

Label synthetic jobs with the teacher prompt. Apply the same normaliser.

**Distribution minimums across entire pool:**

LOCATION (must include these specific locations across training data):
- OUTSIDE_UK: 100 total, covering at least 30 unique countries/cities including: United States, Germany, France, Canada, Australia, India, Singapore, Netherlands, Spain, Ireland (Republic of), Czech Republic, Poland, Sweden, Switzerland, Japan, Brazil, South Africa, UAE, remote US, remote India, New York, San Francisco, Berlin, Paris, Toronto, Sydney, Mumbai, Amsterdam, Tokyo, Dubai
- UK_OTHER: 80 total, covering at least 15 unique UK locations including: Manchester, Bristol, Edinburgh, Glasgow, Cardiff, Birmingham, Leeds, Belfast, Nottingham, Cambridge, Oxford, Liverpool, Brighton, Reading, "United Kingdom" (bare)
- LONDON_OR_REMOTE: 250 total, covering formats: "London", "London, England", "London, England, United Kingdom", "Remote", "remote", "UK Remote", "United Kingdom (Remote)", "London (Remote)", "Fully Remote", "Remote - UK", "London, England, United Kingdom (Hybrid)"
- MISSING: 15 total

TECH:
- NONE: 150 (must include jobs with prominent React, Python, Go, Java, Ruby, .NET, PHP, Rust, Kotlin, C#, Scala — all explicitly ignored in reasoning)
- JS_TS: 100 (cover: "JavaScript", "javascript", "JS", "TypeScript", "typescript", "TS", "JavaScript/TypeScript")
- NODE: 80 (cover: "Node.js", "node.js", "NodeJS", "nodejs", "Nodejs", "Node", "node")
- NODE_JS_TS: 120
- AI_ML: 20
- JS_TS_AI_ML: 20
- NODE_AI_ML: 20
- NODE_JS_TS_AI_ML: 30

COMP:
- NO_GBP with USD visible: 40
- NO_GBP with EUR/AUD/other visible: 20
- NO_GBP with daily rate visible (£X/day, £X per day, £X p/d): 15
- NO_GBP with no salary at all: 120
- NO_GBP with midpoint £45k-£54,999: 10
- UP_TO_ONLY: 40 (cover: "Up to £90k", "up to £120,000", "to £80k", "To £95,000 per annum")
- BELOW_45K: 20
- RANGE_55_74K: 50 (cover: "£55k-£70k", "£60,000-£75,000", "£65k p.a.", "£70,000 per annum")
- RANGE_75_99K: 80
- ABOVE_100K: 100 (cover: "£100k-£130k", "£110,000-£140,000", "£120k+", "£105,000 per annum")

ROLE:
- SENIOR_PLUS standard: 250
- SENIOR_PLUS edge: 40 (cover: "Snr", "Sr.", "Engineer III", "SWE III", "VP of Engineering", "Founding Engineer", "Distinguished Engineer", "Head of Engineering", "Tech Lead", "Principal")
- MID_LEVEL: 120 (cover: "Full Stack Developer", "Full-Stack Engineer", "Fullstack", "Mid-Level", "Midlevel", "Software Engineer II", "SWE II", "Engineer II")
- NO_SENIORITY: 120

LABEL BALANCE (computed by code from semantic tokens):
- good_fit: 200+
- maybe: 300+
- bad_fit: 300+
- Borderline (computed score 45-75): 150+

EDGE CASES (at least 5 examples each):
- "Dublin, Ireland" → OUTSIDE_UK (Ireland ≠ UK)
- "Belfast, Northern Ireland" → UK_OTHER (Northern Ireland = UK)
- "Cork, Ireland" → OUTSIDE_UK
- "Derry, Northern Ireland" → UK_OTHER
- Location says "Remote" but JD says "Remote US only" → LONDON_OR_REMOTE (rule: use location field only)
- Location says "London" but JD says "this role is based in our Berlin office" → LONDON_OR_REMOTE (rule: use location field only)
- Title says "Senior" but JD says "junior/mid level developer" → SENIOR_PLUS (rule: title takes priority)
- Node.js listed as "nice to have" not required → still counts for tech (rule: scan entire JD for keywords)
- "AI-powered company" in company description but no AI/ML in requirements → no AI/ML credit
- "AI experience required" in requirements → AI/ML credit
- Salary in title "Up to £90k" but different salary in JD body → use JD body salary if it has a proper range, otherwise UP_TO_ONLY
- "£500/day" or "£600 per day" or "£550 p/d" → NO_GBP (daily rate)
- "$120,000" and "£80,000-£95,000" both in same JD → use GBP only → RANGE_75_99K
- "React Native" and "React.js" with no Node.js → NONE for tech (React = 0 points)
- "TypeScript AND JavaScript" both mentioned → JS_TS (counted once)
- "node" meaning network/infrastructure node, not Node.js → NONE (context dependent, GPT-4o-mini handles this)
- Job with no description, just a title → tech=NONE, comp=NO_GBP

Cap: total synthetic ≤ 25% of final training set. No single gap category > 60 synthetic jobs.

**1D — Save everything as `all_labeled_pool.jsonl`**

---

### Step 2: Build Eval Set + Training Set

**2A — Select 150 eval jobs**

- 50 good_fit, 50 maybe, 50 bad_fit (labels computed by code from semantic tokens)
- Must include: 15+ OUTSIDE_UK, 10+ comp hard negatives (UP_TO_ONLY, daily rates, USD), 10+ tech NODE/NODE_JS_TS, 5+ edge role titles, 20+ borderline scores (45-75)
- Prefer real jobs over synthetic
- **Manually verify all 50 maybe labels.** Read the JD. Check each semantic token makes sense.
- Spot-check 20 good_fit and 20 bad_fit labels
- Fix any incorrect labels
- Lock as `eval_150_golden.jsonl`. Record SHA-256 hash.

**2B — Assemble ~800 training jobs**

From the remaining pool:
- good_fit 180-200, maybe 280-320, bad_fit 280-320
- Enforce all field minimums from Step 1C tables
- Priority: real jobs → augmented → synthetic
- Cap synthetic at ≤ 25%
- Deduplicate by hash(title + company + location). Verify zero eval overlap.

**2C — Handle long sequences**

For each training example, estimate token count. If total > 7500:
- Scan JD for all occurrences of £, $, €, salary-related keywords (salary, compensation, pay, per annum, p.a.)
- Mark a 100-word window around each occurrence as protected
- Also protect the first 300 words (usually contains role/requirements) and last 200 words (often has salary/benefits)
- Truncate from the middle of the unprotected region
- Insert `[...]` at truncation point
- Log which jobs were truncated

**2D — Format as chat JSONL**

```json
{"messages": [
  {"role": "system", "content": "Respond with JSON only."},
  {"role": "user", "content": "Classify this job. Output JSON only.\n\nTitle: Senior Node.js Engineer\nLocation: London, England, United Kingdom\nDescription: [jd_text]...\n\n{\"reasoning\":\"...\",\"loc\":\"...\",\"role\":\"...\",\"tech\":\"...\",\"comp\":\"...\"}"},
  {"role": "assistant", "content": "{\"reasoning\":\"loc: 'London, England, United Kingdom' → London found → LONDON_OR_REMOTE. role: 'Senior' in title → SENIOR_PLUS. tech: node.js found, typescript found, react ignored → NODE_JS_TS. comp: '£85,000-£120,000' midpoint £102.5k → ABOVE_100K.\",\"loc\":\"LONDON_OR_REMOTE\",\"role\":\"SENIOR_PLUS\",\"tech\":\"NODE_JS_TS\",\"comp\":\"ABOVE_100K\"}"}
]}
```

Split: 720 train / 80 valid (stratified by computed label).

Generate `distribution_report_final.json`. Verify all minimums are met.

---

### Step 3: Train

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B-Instruct-4bit |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Learning rate | 5e-5 |
| Batch size | 2 |
| Grad accumulation | 8 (effective batch = 16) |
| Iters | 600 |
| Warmup | 50 |
| Eval every | 25 |
| Save every | 25 |
| Val batches | 40 |
| mask_prompt | true |
| grad_checkpoint | true |

During training:
- Monitor val loss at every checkpoint
- At iterations 150, 300, 450, 600: run full eval on `eval_150_golden.jsonl` using the code layer to compute score/label from semantic tokens
- Select checkpoint with highest eval label accuracy

If OOM: reduce batch to 1, grad_accum to 16.

---

### Step 4: Evaluate and Ship

Run `eval_finetuned.py` on `eval_150_golden.jsonl` using the best checkpoint.

**Evaluation checks each field independently:**
- loc accuracy: does the student's loc token match the golden loc token?
- role accuracy: does the student's role token match?
- tech accuracy: does the student's tech token match?
- comp accuracy: does the student's comp token match?
- score accuracy: does the code-computed score match the golden code-computed score?
- label accuracy: does the code-computed label match the golden label?

**Reasoning is never evaluated.** It exists only as a generation scaffold.

If the student outputs an invalid token (e.g., `"loc": "LONDN_OR_RMOTE"`), count it as wrong for that field. Track invalid token rate separately — if > 5%, something went wrong with training.

**If label accuracy ≥ 90%:** done.

**If label accuracy < 90%:**
- Analyse which field × token combinations are failing
- Generate 50-100 targeted synthetic jobs for those gaps
- Label, add to training set, retrain at 400 iters
- Re-evaluate
- If still < 90%: consider upgrading to Qwen2.5-1.5B or 3B

---

## Predicted Issues and Mitigations

**The model misspells a semantic token.**
"LONDON_OR_REMOTE" is 6 tokens. The model might output "LONDON_OR_REMOT" or "LONDON_REMOTE".

Mitigation: in the code layer, implement fuzzy matching. If the output doesn't exactly match a legal token, find the closest match by edit distance. If edit distance ≤ 2, use the closest match. If > 2, mark as invalid. After 800 training examples where the model generates each token hundreds of times, misspellings should be < 1%.

**The tech field has 8 categories — too many for 0.5B?**

Mitigation: In practice, 4 categories dominate (NONE, JS_TS, NODE_JS_TS, NODE). The rare ones (AI_ML, NODE_AI_ML, etc.) will have fewer training examples but the reasoning scaffold helps. If post-eval shows the model consistently confuses the AI/ML variants, consider collapsing to 5 categories: NONE, JS_TS, NODE_BASED (10+), NODE_JS_TS_BASED (15+), and FULL_STACK_AI (20+). But try 8 first.

**The minimal student prompt is too minimal.**

Mitigation: the model learns the classification rules from 800 examples. Each example's reasoning explicitly shows the rule: "node.js found → NODE_JS_TS". The model sees this pattern hundreds of times. If post-eval shows systematic blind spots, add one targeted line to the prompt (e.g., "Only £ salaries count, ignore $ € and daily rates"). But start minimal.

**GPT-4o-mini doesn't consistently use the semantic tokens.**

Mitigation: the normaliser after labeling validates every field value against the legal token list. Any output with illegal tokens gets re-labeled. This is a hard constraint, not a soft check.

**The model outputs valid tokens but in wrong fields (e.g., LONDON_OR_REMOTE in the role field).**

Mitigation: the code layer validates each field independently. A role value that isn't in ROLE_MAP throws an error. This should be vanishingly rare after training.

**Training data diversity for location recognition.**

The student can't memorise every city. It learns patterns from data:
- UK indicators: "England", "Scotland", "Wales", "Northern Ireland", "United Kingdom", and specific UK cities seen in training
- Non-UK indicators: country names, US state names, "Remote (US)", non-UK city names seen in training
- The 30+ unique non-UK locations and 15+ unique UK locations in training create enough pattern diversity for generalisation

Remaining risk: a city the model has never seen (e.g., "Tallinn, Estonia"). As long as "Estonia" or any non-UK country name appears, the model should generalise to OUTSIDE_UK. Include a few obscure locations in training to test this.

**Daily rates and contract salaries.**

The teacher prompt explicitly handles these: "£500/day → daily rate ignored → NO_GBP". Training data includes 15+ daily rate examples. The reasoning scaffold primes the model: when it generates "daily rate ignored", the next probable token for comp is NO_GBP.

**"Up to" salary trap.**

40 training examples of UP_TO_ONLY with reasoning like "'Up to £90,000' no lower bound → UP_TO_ONLY". This is one of the most consistent patterns for the model to learn because the reasoning explicitly states the rule.

**Eval set is judged against GPT-4o-mini labels.**

Manual verification of all 50 maybe labels and spot-checking 40 others means ~60% of the eval set is human-verified. The remaining 40% are good_fit/bad_fit labels which are much easier to get right (obvious matches/mismatches). This is sufficient for a reliable benchmark.

**Re-labeling the 450 changes some classifications.**

Expected. The new schema (semantic tokens) and updated rules (Engineer III, daily rates) will change some labels. After labeling, diff old score vs new code-computed score. Log every change. All changes should be improvements.

**Sequence length overflow from reasoning + JD.**

The reasoning field is shorter with semantic tokens (no arithmetic at the end). The student prompt is ~30 tokens (vs ~250 in V4). Combined savings: ~250 tokens per example. Smart truncation (protecting salary windows) handles the remaining long JDs.

---

## Execution Checklist

```
STEP 1: LABEL + FILL GAPS
[ ] 1A: Send ~1553 jobs through GPT-4o-mini batch API with teacher prompt
[ ] 1B: Parse outputs, validate semantic tokens, normalise reasoning format
[ ] 1C: Cross-check reasoning vs token consistency, re-label failures
[ ] 1D: Count distributions, identify gaps against minimums
[ ] 1E: Generate synthetic jobs for each gap with noise/contradictions
[ ] 1F: Post-process synthetic JDs (randomise capitalisation, add noise)
[ ] 1G: Label synthetic jobs, normalise, validate
[ ] 1H: Save all_labeled_pool.jsonl

STEP 2: BUILD EVAL + TRAINING SETS
[ ] 2A: Select 150 eval jobs (50/50/50, gap coverage, prefer real)
[ ] 2B: Manually verify all 50 maybe labels + spot-check 40 others
[ ] 2C: Lock eval_150_golden.jsonl
[ ] 2D: Select ~800 training jobs per distribution targets
[ ] 2E: Enforce minimums, cap synthetic ≤25%
[ ] 2F: Deduplicate, verify zero eval overlap
[ ] 2G: Smart-truncate long JDs (protect salary windows)
[ ] 2H: Format chat JSONL, split 720/80
[ ] 2I: Verify distribution_report_final.json

STEP 3: TRAIN
[ ] 3A: Set LoRA config (rank 16, lr 5e-5, 600 iters)
[ ] 3B: Train, monitor val loss
[ ] 3C: Run eval at iters 150, 300, 450, 600
[ ] 3D: Select best checkpoint by label accuracy

STEP 4: EVALUATE + SHIP
[ ] 4A: Full eval on eval_150_golden.jsonl
[ ] 4B: Score each field independently + code-computed label
[ ] 4C: ≥90% → done
[ ] 4D: <90% → analyse, patch with 50-100 targeted jobs, retrain 400 iters
```

**Total time: 4-6 hours. API cost: ~$1-2.**

---

## What Changed from V4

| Change | Why |
|--------|-----|
| Numeric scores → semantic tokens | 0.5B models classify labels better than picking numbers. "LONDON_OR_REMOTE" is semantically grounded in the JD text. "25" is abstract. |
| Removed score and label from model output | Code computes both deterministically from tokens. Eliminates arithmetic errors and label misclassification. |
| Reasoning no longer includes arithmetic | No more "25+25+15+25=90 → good_fit". Fewer generated tokens, fewer error opportunities. |
| Student prompt reduced to 3 lines | No rules, no examples, no token tables. Model learns everything from training data. |
| Added comp=UP_TO_ONLY as distinct token | Separates "Up to £X" (a trap) from "no salary at all". Model learns the specific rule. |
| Added fuzzy matching in code layer | Handles minor token misspellings without counting as errors. |
| Added NO_GBP subcategory for £45k-£54,999 range | Previously ambiguous. Now explicitly mapped. |
| Added daily rate handling in teacher prompt | "£500/day → daily rate ignored → NO_GBP" is now explicit. |
| Added comprehensive edge case list | Dublin vs Belfast, daily rates, "Remote US", misleading JD locations, etc. |
| Smart truncation protects salary windows | Finds £/$ positions and preserves surrounding context instead of blind positional truncation. |
