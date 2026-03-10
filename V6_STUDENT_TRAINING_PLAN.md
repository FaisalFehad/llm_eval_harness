# V6 Training Plan — Data Quality + Contrastive Retrain

## Summary
Major upgrade from V5.1 (83.9% label accuracy, checkpoint 875) targeting 90-93%. Focus: data quality improvements, dataset pruning, and 140 targeted contrastive training jobs. Fresh LoRA from base. No architecture changes.

## What Changed from V5.1
- **New teacher prompt (teacher_v6.txt)** — engineering gate for ROLE, comprehensive COMP ignore rules, strict reasoning format
  - **[2026-03-09 UPDATE]** Reasoning format redesigned: ~~single `reasoning` string~~ → 8-field interleaved structured format (`loc_reason`, `loc`, `role_reason`, `role`, `tech_reason`, `tech`, `comp_reason`, `comp`). See Step 4b below and Finding 27.
- **[2026-03-09 UPDATE]** V7 teacher prompt architecture: 4 fields → 6 fields (location, work_arrangement, scope, seniority, tech, comp). Richer token vocabulary for better student training signal. Semantic rules replace keyword matching. Backward-compatible via scoring translation layer. See Step 4c below and Finding 28.
- **Rule Contract Lock (Phase 1.5)** — all ambiguous rules resolved BEFORE re-labeling
- **Re-label ALL training data** with V6 teacher prompt at temperature=0 for uniform consistency
- **Deduplicate** training data before re-labeling (4× Engineering Manager duplicates, etc.)
- Audit existing training data for patterns the model can't learn
- Prune ~100-120 trivially easy bad_fit jobs (model already gets bad_fit 96% right)
- 140 contrastive training jobs where same JD has one field changed → different label
- Programmatic variant creation (regex replacement, not GPT rephrasing)
- All labeling at temperature=0 for consistency
- Batch size 1 + grad_accum 16 to prevent OOM crash
- 1000 iters (V5.1 was still improving at 875 when it OOM'd at 890)
- Removed Batch I (format variance — conflicting signals)
- Reduced Batch H (double ambiguity) from 15 to 6
- Expanded Batch J (red herrings) from 15 to 25
- Smaller, cleaner dataset (~890 vs 895) that's higher quality

### Root Causes Identified (V5.1 Analysis)
See `V6_DIAGNOSTIC_FINDINGS.md` for full details on all 20 findings.

1. **Conflicting training labels**: "Engineering Manager" labeled 3 ways (SENIOR_PLUS/NO_SENIORITY/MID_LEVEL) — caused by incomplete V5 teacher prompt keyword list
2. **Duplicate jobs amplifying wrong labels**: 4× "Engineering Manager (React Native)" all labeled NO_SENIORITY (wrong)
3. **Token frequency imbalance**: NONE=54.5% of tech, NO_GBP=64.2% of comp → model defaults to majority token when uncertain
4. **Boundary zone starvation**: Score 65-74 has only 41 jobs (5.4%) but causes ~50% of eval errors
5. **77 non-software "engineer" titles** passing the engineering check incorrectly
6. **Generalization gap**: 8.6% overall (tech=12%, comp=9.5%, good_fit=17%)

---

## Phase 1: Audit and Diagnose

### Step 1: Check labeling temperature history ✅ DONE
- Current temperature in label-jobs.ts: 0 ✅
- File is untracked — no git history to verify past temperature
- **Decision**: Re-label everything with V6 prompt anyway (Step 5.4), so past inconsistency is moot

### Step 2: Run 875 checkpoint on training data ✅ DONE
**Results** (762 training jobs):
```
Label accuracy: 92.5% (train) vs 83.9% (eval) → 8.6% gap
Field gaps: loc=4.5%, role=2.4%, tech=12.0%, comp=9.5%
Per-label: good_fit gap=17%, maybe gap=~8%, bad_fit gap=~2%
```
**Key training errors**: AI_ML→NONE (38), JS_TS→NONE (22), SENIOR_PLUS→NO_SENIORITY (25)
**Interpretation**: Large tech/comp gaps = eval has patterns training doesn't cover. The 25 SENIOR_PLUS errors trace to conflicting "Engineering Manager" labels.

### Step 3: Trace the 24 eval errors from 875 checkpoint ✅ DONE
Error breakdown: 11 good_fit→maybe, 5 maybe→good_fit, 6 maybe→bad_fit, 2 bad_fit→maybe.

**Categorisation**:
- B (batches cover): ~15 errors (tech/comp discrimination, covered by batches A-F)
- C (new gaps): 3-4 errors (pending Batch J slots 22-25)
- D (capacity limits): 7 always-wrong jobs across all checkpoints

**Additional diagnostics run**:
- Checkpoint stability: 73 always correct, 38 oscillating (25%), 7 always wrong
- Boundary zone analysis: Score 65-74 has only 41 jobs (5.4%), causes ~50% of errors
- Token frequency: MID_LEVEL=7.9%, JS_TS_AI_ML=0.9% (7 jobs!)

### Step 4: Count and prune trivially easy bad_fit jobs ✅ DONE
**Found**: 123 jobs matching all criteria (OUTSIDE_UK/MISSING + NO_SENIORITY + NONE + NO_GBP + bad_fit).

Plan: Remove ~100-120 (keep score 30-49 near boundary). Only remove score 0-20.

Expected rebalancing:
```
Before: good_fit 113 (13%), maybe 219 (24%), bad_fit 563 (63%)
After:  good_fit 113 (15%), maybe 219 (29%), bad_fit ~440 (56%)
```

### Step 5: If temperature was previously > 0
Re-label ALL training jobs with temperature=0. Cost: ~$1, time: 15 min.
Even if only some jobs were labeled at higher temperature, re-label everything for uniform consistency.

---

## Phase 1.5: Rule Contract Lock

**Purpose**: Resolve ALL ambiguous classification rules BEFORE re-labeling any data. Changing rules after labeling means re-labeling again.

### Step 5.1: Teacher Prompt V6 (✅ DONE — `prompts/teacher_v6.txt`)

The V5 teacher prompt used fragile keyword lists that caused conflicting labels. The V6 prompt makes three structural changes:

1. **Engineering Gate (ROLE Step 1)**: Instead of a keyword list, the prompt defines engineering roles semantically. "Is this a software/data/platform engineering role?" Gates non-engineering titles to NO_SENIORITY regardless of seniority keywords.

2. **Comprehensive COMP Ignore Rules**: Explicit list of salary-like patterns that are NOT valid GBP annual salaries: TC, OTE, daily rates, package amounts, bonus-inclusive, equity, non-salary £ mentions. Each must appear in reasoning with `→ ignored`.

3. **Strict Reasoning Format**: `loc: ... → TOKEN`, `role: ... → TOKEN`, `tech: ... → TOKEN`, `comp: ... → TOKEN`. No narrative. Factual only.
   - **[2026-03-09 UPDATE]** This single-string format was superseded by 8-field interleaved structured format. See Step 4b in Execution Order. Motivation: 1,048 reasoning abbreviation mismatches found in V5 pool (Finding 27).

**Bugs found and fixed during review**:
- Removed `architect` from SENIOR_PLUS indicators (kept in engineering gate). Prevents "Associate Solutions Architect" → SENIOR_PLUS.
- Added `leader` to SENIOR_PLUS indicators. Ensures "Engineering Team Leader" → SENIOR_PLUS.
- Added `chief` to SENIOR_PLUS indicators. Ensures "Chief Architect" → SENIOR_PLUS.
- MID_LEVEL uses `full stack, full-stack, fullstack` (not "full stack engineer") to catch "Full Stack Developer" too.
- Added explicit AI_ML trigger keyword list (machine learning, ML, deep learning, LLM, NLP, etc.) — was previously unspecified.

### Step 5.2: Identify Spec Violations vs Spec Gaps (✅ DONE)

Reviewed all 24 SENIOR_PLUS jobs with no V5 keyword justification (was estimated at 21, actual count is 24). Found **4 categories**, not 2 — including a new data integrity issue:

| Category | Count | What It Means |
|----------|-------|---------------|
| **SPEC_GAP** | 14 | V5 rules missing manager/director/cto/sr. Labels are correct. V6 prompt fixes. |
| **TITLE_MISMATCH** | 7 | Stored title ≠ title teacher saw at labeling time. Labels are WRONG for stored title. |
| **SPEC_VIOLATION** | 2 | Teacher broke rules. "Data Scientist" (likely also title mismatch), "VP HR" (not engineering). |
| **V6_STILL_WRONG** | 1 | "Sr. Manager, Digital Activation" — V6 keyword matches but fails engineering gate → NO_SENIORITY. |

**Spec gaps (14)**: V5 keyword list was incomplete. These labels are correct — the V6 prompt now includes manager (7), director (6), cto (1), sr (1).

**Title mismatches (7)**: The stored title differs from the title the teacher saw at labeling time. The reasoning field reveals the original title:
- "Node.JS Engineer" ×4 — teacher saw "Senior Backend Developer" → SENIOR_PLUS (correct for original, WRONG for stored)
- "Software Developer (remote)" — teacher saw "Senior Full-Stack Engineer"
- "Machine Learning Engineer (all)" — teacher saw "Senior Machine Learning Engineer"
- "DevOps Engineer" — teacher saw "Senior..."

These teach the student that plain "Engineer" = SENIOR_PLUS. Root cause: titles were modified after labeling, or labeling script used a different title field. **Needs investigation before re-labeling** — see Finding 19 in V6_DIAGNOSTIC_FINDINGS.md.

**Additional flags**:
- "Regional Director" (line 632) — classified as spec gap (director), but NOT software engineering. V6 gate should → NO_SENIORITY.
- "Director Fire Engineering" (line 145) — fire engineering, NOT software. V6 gate should → NO_SENIORITY.
- 4× "Node.JS Engineer" are duplicates amplifying the wrong title-mismatch signal.

### Step 5.3: Deduplicate + Audit Training Data (✅ DONE)

**Why this was needed**: The original plan called for simple deduplication (remove 4× Engineering Manager, etc.). But investigation revealed the problem was much bigger:
- No unified data quality tooling existed — 8 scattered scripts each silently dropped bad data with no audit trail or quarantine
- Finding 8 showed duplicates actively harming training (4× wrong label = 4× wrong gradient)
- The 895-job training set had never been systematically audited — unknown quality issues hiding
- The upcoming V6 re-labeling needs clean input — re-labeling duplicates, eval-contaminated jobs, and trivially easy bad_fit wastes API cost and training time on data that either hurts or teaches nothing
- This was issue 3 of the 6 pre-training issues identified during V5.1 diagnostics

Built a comprehensive data audit pipeline (`src/cli/audit-training-data.ts`) that runs 6 critical checks and 7 warning checks in a single pass.

**Audit script built with modes:**
- Default: audit-only (report + exit code)
- `--dry-run`: report without exit codes (used as pre-flight gate)
- `--clean --output <path>`: produce cleaned file, quarantine removed jobs
- `--remove-suspicious`, `--remove-trivial`: optional aggressive cleaning
- Quarantine output: `<input-dir>/quarantine/{duplicates,bad_data,suspicious}.jsonl`

**6 Critical checks** (block pipeline if any found):
1. Missing required fields (job_id, title, location, jd_text, loc, role, tech, comp, label)
2. Invalid token values (not in allowed vocabulary)
3. Score-token mismatch (token scores don't match stored scores)
4. Label-score mismatch (label doesn't match score thresholds)
5. Eval contamination (3 strategies: job_id, title+company key, JD fingerprint SHA-256)
6. Malformed JSON / unparseable records

**7 Warning checks** (report + quarantine, don't block):
1. Duplicate detection (job_id, title+company, JD fingerprint — with contrastive pair awareness)
2. Short JDs (< 100 chars)
3. Long JDs (> 10,000 chars)
4. HTML artifacts in JD text
5. Title mismatches (Finding 19 — stored title doesn't match reasoning)
6. Suspicious token combos (non-engineering titles with SENIOR_PLUS)
7. Source-weight cap (no single source > 5%)

**Results on train_860.jsonl (895 jobs):**
```
Critical issues: 17
  - Eval contamination: 17 (synthetic jobs sharing title+company with eval set)
Warnings: 241
  - Duplicates: 107 (by title+company and JD fingerprint)
  - Contrastive pairs protected: 19 (share source_job_id — intentional variants)
  - Suspicious tokens: 24 (non-engineering SENIOR_PLUS)
  - Title mismatches: 7 (Finding 19)
  - Source weight cap: 4
  - Long JDs: 3
  - HTML artifacts: 1
```

**Clean mode results (train_860.jsonl → train_clean.jsonl):**
```
Input: 762 jobs (after eval contamination removed)
Kept: 526 jobs
Removed: 236 total
  - Eval contamination: 17
  - Duplicates: 107
  - Trivial bad_fit: 123 (OUTSIDE_UK/MISSING + NO_SENIORITY + NONE + NO_GBP)
  - Suspicious: 35 kept as warnings (not removed by default)
```

**Key design decisions:**
- Contrastive pairs (jobs sharing `source_job_id`) are detected but NOT removed — they're intentional variants
- Trivially easy bad_fit removal requires explicit `--remove-trivial` flag
- Audit is wired into `build-datasets.ts` as a pre-flight gate (runs automatically before dataset builds)

**Original dedup plan targets confirmed:**
- 4× "Engineering Manager (React Native)" duplicates: caught by title+company dedup
- 4× "Node.JS Engineer" synthetic variants: caught by JD fingerprint dedup
- Source-weight cap: 4 sources over 5% threshold flagged

### Step 5.3b: Build Validation Pipeline (⬜ TODO)

**Why this was needed**: V5 and V5.1 had 9 data quality issues that were ALL discovered AFTER training — conflicting labels, duplicates, title mismatches, eval contamination, trivially easy bad_fit, token imbalance, boundary starvation, non-engineering titles, and 17 contaminated synthetic jobs. None were caught before training because no validation existed between labeling and training. Two full training runs (~4h each) produced below-target results, and every root cause was a data issue that automated checks could have caught.

**Solution**: Two-gate validation system bracketing the labeling step:

```
Input pool → PRE-LABEL AUDIT → Label clean jobs → POST-LABEL AUDIT → Build datasets → Train
               (instant)          (~12 min)          (instant)         (instant)       (hours)
```

**Implementation (4 items, ~65 lines across 2 existing files + 1 config):**

1. **`--pre-label` mode in `audit-training-data.ts`** (~20 lines) — runs on unlabeled input data:
   - Duplicates (don't waste API calls labeling the same job twice)
   - Eval contamination (don't label eval jobs)
   - JD-embedded title mismatch (flag jobs where JD body contains a senior title not in the Title: field)
   - Short/corrupt JDs
   - Trivially easy bad_fit (don't label jobs we'll prune anyway)

2. **3 new post-label checks in `audit-training-data.ts`** (~30 lines):
   - Title-echo: reasoning references a title not matching the stored title
   - Distribution balance: warn if any token > 50% of its field, or boundary zone (score 50-74) < 15%
   - Label consistency: same title should always produce the same role token

3. **Per-run log file in `label-jobs.ts`** (~15 lines):
   - Timestamped JSONL log of every request/response
   - Saved to `data/v5/labeling_runs/YYYY-MM-DD_HHMMSS.log.jsonl`
   - For future debugging if training results are unexpected

4. **Promptfoo test config for V6 teacher prompt** (YAML file):
   - Tests against ~25-30 edge cases covering every finding: title mismatches, Engineering Manager, spec gaps, non-engineering titles, architect/leader/chief, AI/ML discrimination, COMP ignore rules (TC/OTE/daily/package)
   - Quick prompt sanity check (5 seconds) before any labeling run
   - **This is the highest-ROI component** — 8 of the 13 V5/V5.1 issues were prompt problems that Promptfoo would have caught in 5 seconds before a single job was labeled (see Finding 25 retrospective). The 5-second test would have saved ~8h of training and weeks of analysis.

**What we deliberately did NOT build (and why):**
- No `--sample N` mode — Promptfoo + pre-label audit make it redundant
- No resume mode in labeling — labeling rarely fails midway (retries handle rate limits)
- No Instructor (retry-based validation) — designed for capable models, not 0.5B student
- No constrained decoding — 0.7% invalid token rate doesn't justify MLX integration complexity
- No separate validation script — extends existing tools instead of adding new ones

**Verification**: Every V5/V5.1 issue mapped against the two-gate system — all 9 caught. Zero gaps.

**Known limitations:**
- Pre-label JD title detection is heuristic (regex) — will have false positives/negatives. Flags for human review, not auto-reject.
- Post-label title-echo parses GPT's reasoning format — breaks if GPT changes its phrasing.
- Distribution thresholds (50% token cap, 15% boundary zone) are educated guesses from V5/V5.1 experience, not proven. May need tuning.
- Only catches KNOWN issue types — if V6 introduces a new problem we haven't seen, neither gate catches it.
- Manual correction of title-mismatch jobs is a patch — new data with the same LinkedIn quirk will need the same treatment.
- V6 prompt may not fix GPT's title adherence — we detect the problem, we don't eliminate the root cause.
- Promptfoo tests ~20 edge cases, not all 900 jobs — post-label audit is the real safety net.

### Promptfoo V6 Test Cases (for Step 5.3b implementation)

These are the ~28 test cases the Promptfoo config should cover. Each tests a specific finding or edge case. The assertion checks that the V6 prompt produces the expected token.

**ROLE — Engineering gate + seniority (Findings 6, 7, 10, 11, 12, 18, 20):**

| # | Title | Expected role | Tests |
|---|---|---|---|
| 1 | Engineering Manager | SENIOR_PLUS | Finding 6: "manager" now in keyword list |
| 2 | Director of Engineering | SENIOR_PLUS | Finding 7: spec gap — "director" |
| 3 | CTO (Chief Technology Officer) | SENIOR_PLUS | Finding 7: spec gap — "cto" |
| 4 | Sr. Software Engineer | SENIOR_PLUS | Finding 7: spec gap — "sr" |
| 5 | Engineering Team Leader | SENIOR_PLUS | Finding 12: "leader" keyword |
| 6 | Chief Architect | SENIOR_PLUS | Finding 18: "chief" keyword |
| 7 | Founding Engineer | SENIOR_PLUS | Edge case: "founding" |
| 8 | Engineer III | SENIOR_PLUS | Edge case: Roman numeral seniority |
| 9 | Associate Solutions Architect | NOT SENIOR_PLUS | Finding 11: "architect" removed from SENIOR_PLUS |
| 10 | Regional Director | NO_SENIORITY | Finding 20: not engineering |
| 11 | Director Fire Engineering | NO_SENIORITY | Finding 20: fire engineering ≠ software |
| 12 | Senior Marketing Manager | NO_SENIORITY | Finding 10: non-engineering with "Senior" |
| 13 | Sales Engineer | NO_SENIORITY | Finding 10: non-engineering "engineer" |
| 14 | VP of Human Resources | NO_SENIORITY | Finding 10: non-engineering with "VP" |
| 15 | Full Stack Developer | MID_LEVEL | Basic mid-level test |

**ROLE — Title-mismatch adherence (Finding 19/24):**

| # | Title (field) | JD body says | Expected | Tests |
|---|---|---|---|---|
| 16 | Node.JS Engineer | "Senior Backend Developer" | NOT SENIOR_PLUS | GPT must use Title: field, not JD |
| 17 | Software Developer (remote) | "Senior Full-Stack Engineer" | NOT SENIOR_PLUS | Same adherence test |
| 18 | Data Scientist | "Data Scientist III" | NOT SENIOR_PLUS | Same adherence test |

**TECH — AI/ML discrimination (Finding 17):**

| # | Scenario | Expected tech | Tests |
|---|---|---|---|
| 19 | "ML experience required" in requirements | AI_ML | Required = counts |
| 20 | "We are an AI-powered company" in company desc | NOT AI_ML | Company desc = doesn't count |
| 21 | "ML experience is a plus" in nice-to-have | NOT AI_ML | Nice-to-have = doesn't count |

**COMP — Ignore rules (Batch F/J patterns):**

| # | Salary text | Expected comp | Tests |
|---|---|---|---|
| 22 | "£85,000 - £95,000 per annum" | RANGE_75_99K | Basic GBP range |
| 23 | "TC £120,000" | NO_GBP | Total compensation ignored |
| 24 | "OTE £110,000" | NO_GBP | On-target earnings ignored |
| 25 | "£500/day" | NO_GBP | Daily rate ignored |
| 26 | "$110,000 - $125,000" (London job) | NO_GBP | USD ignored |
| 27 | "Total package £130,000 including benefits" | NO_GBP | Package ignored |
| 28 | "We manage £2bn in assets" | NO_GBP | Non-salary £ ignored |

### Step 5.4: Re-label ALL Training Data with V7 Teacher Prompt (⬜ TODO)

**Pre-requisites**: Steps 5.1-5.3b complete. Pre-label audit passes on input data. **All scripts updated for V7 token vocabulary** (Step 5 in Execution Order).

> **[2026-03-09 UPDATE]** Additional pre-requisites: Step 4b (structured reasoning format), Step 4c (V7 architecture), and Step 5 (V7 script updates) must all be complete. Output will use V7's 12-field format (6 _reason + 6 token fields). Also includes new scraped data from Step 4 (distribution analysis): 2,589 jobs in `data/v6/scraped_clean_for_labeling.jsonl`. Student prompt will be `prompts/student_v7.txt`.

After all pre-requisites, re-label the entire (deduplicated, pre-audited) training set with `prompts/teacher_v7.txt` at temperature=0. This:
- Fixes the 10 conflicting "Engineering Manager" labels
- Fixes the 14 spec gap labels (manager/director/cto/sr now in V6 keywords)
- Should fix 5 title-mismatch labels (V6 prompt is more structured, but GPT may still read JD body — verify in Step 5.5D, manually correct if needed)
- Should fix 3 non-engineering titles via engineering gate (uncertain — verify in Step 5.5C)
- Ensures uniform labeling rules across all jobs
- Cost: ~$0.80 (reduced — pre-label audit removes ~240 bad/duplicate jobs before labeling)
- Time: ~12 min
- **Log file saved to `data/v5/labeling_runs/` for future debugging**

### Step 5.5: Verify Re-labeling (⬜ TODO)

After re-labeling, run these SPECIFIC checks. These are not optional — each verifies a claim we're making about V6 but haven't proven yet.

**A. General checks:**
- Count how many labels changed overall
- If > 30 labels change, investigate the biggest shifts before proceeding
- Spot-check 10 random changed labels

**B. Verify spec gap fixes (14 jobs) — V6 keywords work as expected:**
- [ ] All "Engineering Manager" variants → SENIOR_PLUS (lines 14, 119, 287, 530, 741, 751)
- [ ] "Director Engineering" (line 16) → SENIOR_PLUS
- [ ] "Director Of Engineering" (line 386) → SENIOR_PLUS
- [ ] "CTO (chief Technology Officer)" (line 416) → SENIOR_PLUS
- [ ] "Sr. Software Engineer" (line 677) → SENIOR_PLUS
- [ ] "Sr Director Of Engineering" (line 150) → SENIOR_PLUS

**C. Verify engineering gate rejects non-engineering (UNCERTAIN — we are guessing these work):**
- [ ] "Regional Director" (line 632) → NO_SENIORITY (explicitly listed as non-engineering example)
- [ ] "Director Fire Engineering" (line 145) → NO_SENIORITY (fire engineering ≠ software)
- [ ] "Sr. Manager, Digital Activation - Australia" (line 643) → NO_SENIORITY (marketing, not engineering)
- [ ] "Vice President Of Human Resources" (line 536) → NO_SENIORITY (HR, not engineering)
- [ ] Check any "Sales Engineer", "Support Engineer" titles → NO_SENIORITY

**D. Verify title mismatches (8 jobs — no longer a BLOCKER, now a verification gate):**

Root cause investigated (Finding 24): pipeline is clean — GPT-4o-mini reads the JD body title despite "check title only" instruction. This is a prompt adherence issue, not a pipeline bug. Pre-label audit flags these. Post-label audit title-echo check catches them. If GPT still gets them wrong, manually correct the 5 poisonous jobs.

- [ ] "Node.JS Engineer" (4 copies) → check if reasoning references stored title. If still says "Senior Backend Developer" → manually set role to MID_LEVEL (fallback: JD says "Senior")
- [ ] "Software Developer (remote)" → same check
- [ ] "Machine Learning Engineer (all)" → same check
- [ ] "DevOps Engineer" → same check
- [ ] "Data Scientist" → same check (teacher may reference "Data Scientist III" from JD)
- [ ] The post-label audit title-echo check should flag these automatically

**E. Consistency check (now automated in post-label audit):**
- [ ] No title should have more than 1 unique role token across all occurrences (e.g., all "Engineering Manager" must be the same)
- [ ] Count SENIOR_PLUS without V6 keyword → should be 0 (all justified)
- [ ] Distribution balance: no token > 50% of its field, boundary zone (score 50-74) has ≥ 15% of jobs

**If ANY check in section C fails, do NOT proceed to training.** Fix the issue first.
**If checks in section D fail**, manually correct the specific jobs (known, finite list) and proceed.

### Step 5.6: Pre-train Gates (⬜ TODO)

Before starting training, verify these hard gates. If ANY fail, STOP and fix:

1. **No duplicate JDs** in training set (hash check)
2. **Zero overlap** between training and eval sets
3. **"Engineering Manager" labels uniform** (all same role token)
4. **good_fit > 12%**, maybe > 25%, bad_fit < 60% of total
5. **No single source > 5%** of total dataset
6. **Every contrastive variant**: changed field has different token, unchanged fields have same tokens

---

## Phase 2: Generate Contrastive Training Data

### Key Principles
1. **Programmatic variant creation** — regex/string replacement, NOT GPT rephrasing. Unchanged parts must be byte-for-byte identical.
2. **Temperature=0 for all labeling** — consistency over creativity.
3. **Temperature=0.7 for generating base JDs** — variety needed for content.
4. **Each base JD maximally different** — different industry, writing style, length, company type.
5. **Vary salary amounts** — don't repeat £85k-£95k across batches. Use £60k-£70k, £75k-£85k, £80k-£95k, £90k-£100k, £95k-£110k.
6. **Max 3 variants per base JD** — limits memorisation risk.

### Batch A — Comp Amount Contrastive (5 bases × 3 variants = 15 jobs)
Each base has loc+role+tech scoring 50-65 (borderline). Only comp changes.
- Base: real GBP salary range (vary amounts across the 5 bases)
- Variant B: replace salary with "Up to £X" format → UP_TO_ONLY
- Variant C: remove salary text entirely → NO_GBP

Teaches: comp is the label tiebreaker on borderline jobs.

### Batch B — Tech Presence Contrastive (5 bases × 3 variants = 15 jobs)
Each base has loc+role+comp scoring 50-65. Only tech changes.
- Base: Node.js + TypeScript in requirements → NODE_JS_TS
- Variant B: replace with Python + Django → NONE
- Variant C: replace with PHP + Laravel → NONE

Teaches: tech presence determines the token, not tech prominence.

### Batch C — Location Contrastive (5 bases × 3 variants + 4 Remote formats = 19 jobs)
Same JD body, only location field changes.
- Base: "London, England, United Kingdom" → LONDON_OR_REMOTE
- Variant B: UK city (Manchester, Bristol, Edinburgh, Birmingham, Leeds) → UK_OTHER
- Variant C: non-UK (Dublin Ireland, Berlin Germany, Toronto Canada, Amsterdam Netherlands, Sydney Australia) → OUTSIDE_UK

Plus 4 extra jobs with unusual Remote formats:
- "Fully remote (UK)" → LONDON_OR_REMOTE
- "Work from anywhere in UK" → LONDON_OR_REMOTE
- "Remote within United Kingdom" → LONDON_OR_REMOTE
- "UK - Remote / Hybrid" → LONDON_OR_REMOTE

Teaches: location field determines loc. Includes Ireland (not UK) vs Northern Ireland (UK).

### Batch D — AI/ML Discrimination (5 bases × 3 variants = 15 jobs)
Same JD, AI/ML mention changes context.
- Base: "Experience with LLMs and ML pipelines required" in requirements section → AI_ML credit
- Variant B: "We are an AI-powered company" in company description only → NO AI_ML credit
- Variant C: "ML experience is a plus but not required" in nice-to-have → NO AI_ML credit

Teaches: only explicit requirements count for AI_ML token.

### Batch E — Role Title Contrastive (5 bases × 3 variants + 5 edge titles = 20 jobs)
Same JD body (no seniority mentioned in body), only title changes.
- Base: "Senior Software Engineer" → SENIOR_PLUS
- Variant B: "Software Engineer" / "Backend Developer" / "Platform Engineer" → NO_SENIORITY
- Variant C: "Full Stack Engineer" / "Mid-Level Software Engineer" / "Software Engineer II" → MID_LEVEL

Plus 5 edge case titles (same JD pattern, senior Node.js London £80k+):
- "Snr Software Developer" → SENIOR_PLUS
- "Engineer III" → SENIOR_PLUS
- "Founding Engineer" → SENIOR_PLUS
- "Staff Backend Engineer" → SENIOR_PLUS
- "Principal Platform Engineer" → SENIOR_PLUS

Teaches: title keywords determine role, nothing else.

### Batch F — Comp Format + Currency Contrastive (5 bases × 3 variants = 15 jobs)
Same role/location/tech, salary format changes.
- Base: GBP salary range (e.g., "£85,000 - £95,000 per annum") → RANGE_75_99K
- Variant B: USD salary (e.g., "$110,000 - $125,000") in London job → NO_GBP
- Variant C: rotate through: "TC £120,000", "OTE £110,000", "£500/day", "circa £90k including bonus", "Competitive salary + equity" → NO_GBP

Teaches: only GBP base salary ranges count. USD, TC, OTE, daily rates = NO_GBP.

### Batch G — Location MISSING (10 jobs)
Garbage location fields with mixed other fields:
1. "" (empty string)
2. "Not specified"
3. "See description"
4. "N/A"
5. "About Us: We are a leading technology company"
6. "Remote" → this one should be LONDON_OR_REMOTE, not MISSING
7. "TBC"
8. "Various locations"
9. "   " (whitespace only)
10. "Apply for details"

Some JDs mention UK cities, some mention US cities. Model must learn: garbage location = MISSING regardless of JD content.

### Batch H — Double Ambiguity (2 bases × 3 variants = 6 jobs)
Both tech AND comp change simultaneously.
- Base: Node.js + TypeScript + "£90,000-£110,000" → NODE_JS_TS + ABOVE_100K → good_fit
- Variant B: Python + FastAPI + no salary → NONE + NO_GBP → maybe
- Variant C: Go + gRPC + "Up to £90,000" → NONE + UP_TO_ONLY → maybe

Teaches: model must discriminate multiple fields at once.

### Batch J — Red Herrings and Traps (25 independent jobs)
Each teaches a specific real-world trap:

1. Node.js in title but JD says migrating away from Node.js (still counts — rules say scan for presence)
2. "Competitive salary" / "excellent compensation" with no number → NO_GBP
3. "Our clients earn £100k-£150k" — client salary not job salary → NO_GBP
4. Location says London, company section mentions New York HQ → LONDON_OR_REMOTE (location field wins)
5. "TC £150,000" → NO_GBP (total compensation)
6. "OTE £120,000" → NO_GBP (on-target earnings)
7. "£500/day contract rate" → NO_GBP (daily rate)
8. "£450 per day" → NO_GBP (daily rate)
9. "Total package £130,000 including benefits" → NO_GBP
10. "AI" mentioned 5 times in company description but role requires Python/SQL only → tech=NONE
11. Both "£80,000-£100,000" AND "$120,000-$140,000" in same JD, London role → RANGE_75_99K (GBP only)
12. "Engineering Team Leader" — "Leader" not "Lead" → test teacher interpretation
13. "2+ years experience" in JD but "Senior Engineer" in title → SENIOR_PLUS (title wins)
14. "JavaScript" mentioned once in legacy context, role is Python/Django → test teacher interpretation
15. "Greater London Area, United Kingdom" → LONDON_OR_REMOTE
16. "Package £140k" → NO_GBP
17. "Total comp circa £120k" → NO_GBP
18. "£600 p/d" → NO_GBP (daily rate)
19. "Day rate: £550" → NO_GBP (daily rate)
20. "Market-leading compensation" with no number → NO_GBP
21. "We manage £2bn in assets" — £ in non-salary context → NO_GBP
22-25. **Reserved** — filled after Step 3 error analysis with category C patterns

### Removed Batch
- ~~Batch I (Format Variance)~~ — GPT-4o-mini labeling produces inconsistent tokens for reformatted text, teaching conflicting signals. Existing 895 real jobs already provide natural format diversity.

### Batch Summary
| Batch | Description | Count |
|-------|-------------|-------|
| A | Comp amount contrastive | 15 |
| B | Tech presence contrastive | 15 |
| C | Location + Remote formats | 19 |
| D | AI/ML discrimination | 15 |
| E | Role title + edge cases | 20 |
| F | Comp format + currency (TC/OTE/daily) | 15 |
| G | Location MISSING | 10 |
| H | Double ambiguity | 6 |
| J | Red herrings and traps | 25 |
| **Total** | | **140** |

---

## Phase 3: Assemble and Retrain

### Execution Order (Reordered Priorities)

The original plan had phases in logical order but not priority order. This is the actual execution sequence — follow it step by step:

1. ✅ **Lock rules** — Finalise teacher_v6.txt prompt. No rule changes after this point.
2. ✅ **Classify spec violations vs gaps** — Review 24 unjustified SENIOR_PLUS jobs (was estimated 21). Found 14 spec gaps, 7 title mismatches, 2 spec violations, 1 V6-still-wrong.
3. ✅ **Build audit tooling + deduplicate training data** — Built `audit-training-data.ts` with 6 critical + 7 warning checks, contrastive pair awareness, clean mode. Found 17 eval contaminations, 107 duplicates, 123 trivially easy bad_fit. Wired as pre-flight gate into `build-datasets.ts`. Also: 4-layer data loss safeguards, script renames, debris cleanup.
3b. ✅ **Build validation pipeline** — [2026-03-09] Completed: Pre-label audit (`--pre-label` flag), 3 new post-label checks (title-echo, distribution, label consistency), per-run logging in label-jobs.ts, Promptfoo test config (~28 edge case tests). Catches all 9 known V5/V5.1 issue types.
4. ✅ **Distribution analysis & gap filling** — [2026-03-09] Counted token distributions across all_labeled_pool (1,522 jobs) against V5 plan minimums. Found critical gaps: NODE (38 vs 80 min), NODE_AI_ML (0 vs 15 min), good_fit (91 vs 200 min). Searched ~5,000 scraped jobs across all sources. After dedup+quality filtering: 2,543 real + 46 synthetic variants (32 NODE + 14 NODE_AI_ML via tech-swap). Output: `data/v6/scraped_clean_for_labeling.jsonl` (2,589 jobs). Edge case coverage verified (all ≥5 minimum met).
4b. ✅ **Structured reasoning format redesign** — [2026-03-09] Replaced 5-field output (single `reasoning` string + 4 tokens) with 8-field interleaved: `{loc_reason, loc, role_reason, role, tech_reason, tech, comp_reason, comp}`. Prompted by 1,048 reasoning abbreviation mismatches in V5 pool (Finding 27). Teacher, student training, and student inference all use same 8-field format. Eval scores only 4 token fields. 3-layer safeguards built. Files: `teacher_v6.txt`, `student_v6.txt` (new), `audit-training-data.ts`, `format-for-mlx.ts`.
4c. ✅ **V7 teacher prompt architecture** — [2026-03-09] Major redesign of teacher prompt expanding from 4 fields to 6 fields with richer token vocabulary. See Finding 28.
    - **New fields**: `scope` (IN_SCOPE/OUT_OF_SCOPE) and `work_arrangement` (REMOTE/HYBRID/IN_OFFICE/UNKNOWN)
    - **Renamed fields**: `loc` → `location` (tokens: IN_LONDON/FULLY_REMOTE/UK_OTHER/OUTSIDE_UK/UNKNOWN), `role` → `seniority` (tokens: LEVEL_1/LEVEL_2/LEVEL_3)
    - **Split LONDON_OR_REMOTE** into IN_LONDON + FULLY_REMOTE (teaches student finer-grained location patterns)
    - **Split old ROLE** into scope (binary gate) + seniority (3 levels) — cleaner separation of concerns
    - **Semantic rules** replace keyword lists — GPT-4o-mini reasons about intent, not pattern matching
    - **12 JSON keys** in output (6 _reason + 6 token fields)
    - **Scoring**: backward-compatible via translation layer in eval_student.py (V7 tokens → V6 scores → same label formula)
    - File: `prompts/teacher_v7.txt` (107 lines, 5 worked examples)

5. ✅ **Create V7 versions of scripts and prompts** — [2026-03-09] All 10 V7 files created alongside V6 (non-destructive, Critical Rule #12). V6 scripts untouched.
    - `finetune/semantic_tokens_v7.py` ✅ — 6 token tuples, 4 score maps, scope gate, V7→V6 translation. 11 tests passed.
    - `src/lib/semantic-tokens-v7.ts` ✅ — TypeScript types, computeFromTokens(), crossCheckReasoning(). 7 tests passed.
    - `prompts/student_v7.txt` ✅ — 14 lines, 6 field token lists, 12-key JSON template. 8 verification checks.
    - `src/cli/label-jobs-v7.ts` ✅ — V7 labeling with `job_location` collision fix, 6-field distribution reporting. Type-checks clean.
    - `configs/promptfoo_teacher_v7.yaml` ✅ — 36 tests (V6 had 28). New: SCOPE(15), LOC(4), WA(4) categories.
    - `finetune/eval_student_v7.py` ✅ — 6-field comparison, scope gate scoring, V6-compatible quick-copy summary. 4 tests passed.
    - `src/cli/format-for-mlx-v7.ts` ✅ — 12-field V7Job type, `job_location` for raw location. Type-checks clean.
    - `src/cli/audit-training-data-v7.ts` ✅ — V7 tokens, scope gate scoring, scope+seniority consistency checks. Same tsconfig warnings as V6.
    - `finetune/lora_config_v7.yaml` ✅ — V7 data/adapter paths, batch_size=1, all assertions passed.
    - `src/cli/generate-synthetic-hij.ts` — Left as-is (V7 contrastive batches will need V7-aware scripts if needed).
    - `src/cli/check-distribution.ts` — Left as-is (V7 distribution checks via V7 audit script).

5b. ✅ **V7 distribution gap analysis & supplementary data** — [2026-03-09] Analyzed V7 token distribution against V5 plan minimums. Found 3 critical gaps caused by V7's new token vocabulary. Scanned all external sources (job_searcher, job_search_agent_v2) — almost all data already in pool from Step 4. Created 115 supplementary variants from real JDs. See Finding 29.
    - **FULLY_REMOTE gap (CRITICAL)**: V7 split LONDON_OR_REMOTE into IN_LONDON + FULLY_REMOTE. Estimated ~30 FULLY_REMOTE in existing pool vs 60 minimum. LinkedIn scrapes don't capture "Remote" in location field. Zero real remote jobs in any external source. **Fix**: Created 60 location-swap variants from real UK engineering JDs (12 remote formats × 5 donors each).
    - **NODE gap (persistent)**: ~37 NODE-only jobs vs 80 minimum. V6 had 32 synthetic NODE variants but still short. **Fix**: Created 45 additional NODE tech-swap variants from different Python/Java/Go donors.
    - **NODE_AI_ML gap**: ~15 estimated vs 20 minimum. **Fix**: Created 10 NODE_AI_ML tech-swap variants (NODE + AI/ML requirement text appended).
    - **External data scan**: job_searcher (150 custom_training + 1330 balanced_dataset = all dups), job_search_agent_v2 (980 real_linkedin + 317 linkedinScraper = 3 new). Only 3 genuinely new real jobs found.
    - **Files created**: `data/v7/remote_variants.jsonl` (60), `data/v7/synthetic/node_variants_v7.jsonl` (45), `data/v7/synthetic/node_ai_ml_variants_v7.jsonl` (10), `data/v7/gap_fill_real.jsonl` (3)
    - **Totals**: Grand total for labeling = 4,226 jobs (1,522 pool + 2,589 scraped + 115 V7 supplementary). Synthetic total = 161/4,226 = 3.8% (well under 25% cap).

### V7 Distribution Minimums (Updated from V5 Plan for 6-field Architecture)

> **[2026-03-09 UPDATE]** V5 plan had minimums for 4 fields (loc, role, tech, comp). V7 introduces 2 new fields (work_arrangement, scope) and splits loc/role. These are the V7-adjusted targets for the combined labeling pool (~4,226 jobs). Original V5 minimums preserved where fields are unchanged.

**LOCATION (5 tokens)** — V5's LONDON_OR_REMOTE (250 min) splits into IN_LONDON + FULLY_REMOTE:
| Token | V7 Minimum | V5 Equivalent | Notes |
|-------|-----------|---------------|-------|
| IN_LONDON | 200 | was part of LONDON_OR_REMOTE (250) | ~91% of old LONDON_OR_REMOTE pool |
| FULLY_REMOTE | 60 | was part of LONDON_OR_REMOTE (250) | Critical gap — 60 synthetic variants created |
| UK_OTHER | 80 | 80 (unchanged) | |
| OUTSIDE_UK | 100 | 100 (unchanged) | |
| UNKNOWN | 15 | was MISSING (15) | Renamed only |

**WORK_ARRANGEMENT (4 tokens)** — NEW field, no V5 baseline:
| Token | V7 Minimum | Notes |
|-------|-----------|-------|
| REMOTE | 60 | Overlaps with FULLY_REMOTE location jobs, plus remote roles in London |
| HYBRID | 80 | Very common in UK job market, should be well-represented in real data |
| IN_OFFICE | 50 | Less common post-COVID but still present |
| UNKNOWN | 100 | Many JDs don't specify, expect this to be the largest bucket |

**SCOPE (2 tokens)** — NEW field, split from V5's ROLE:
| Token | V7 Minimum | Notes |
|-------|-----------|-------|
| IN_SCOPE | 400 | Engineering roles — majority of training data |
| OUT_OF_SCOPE | 80 | Non-engineering roles (marketing, sales, HR, etc.) |

**SENIORITY (3 tokens)** — Renamed from V5's ROLE:
| Token | V7 Minimum | V5 Equivalent | Notes |
|-------|-----------|---------------|-------|
| LEVEL_3 | 200 | SENIOR_PLUS standard (250) + edge (40) | Combined; V7 semantic rules replace keyword lists |
| LEVEL_2 | 120 | MID_LEVEL (120) | Unchanged target |
| LEVEL_1 | 100 | NO_SENIORITY (120) | Slightly reduced — many out-of-scope jobs will have LEVEL_1 |

**TECH (8 tokens)** — Unchanged from V5:
| Token | V7 Minimum | Notes |
|-------|-----------|-------|
| NONE | 150 | |
| JS_TS | 100 | |
| NODE | 80 | 45 new synthetic variants created |
| NODE_JS_TS | 120 | |
| AI_ML | 20 | |
| JS_TS_AI_ML | 20 | |
| NODE_AI_ML | 20 | 10 new synthetic variants created |
| NODE_JS_TS_AI_ML | 30 | |

**COMP (6 tokens)** — Unchanged from V5:
| Token | V7 Minimum | Notes |
|-------|-----------|-------|
| NO_GBP | 195 | Sum of V5 subcategories (USD:40 + EUR:20 + daily:15 + none:120) |
| UP_TO_ONLY | 40 | |
| BELOW_45K | 20 | |
| RANGE_55_74K | 50 | |
| RANGE_75_99K | 80 | |
| ABOVE_100K | 100 | |

**LABEL BALANCE** — Unchanged from V5:
| Label | V7 Minimum | Notes |
|-------|-----------|-------|
| good_fit | 200+ | |
| maybe | 300+ | |
| bad_fit | 300+ | |
| Borderline (score 45-75) | 150+ | |

> **Note**: These minimums are targets for the FULL labeled pool (all ~4,226 jobs), not the final training set. The training set will be a subset after eval set extraction and pruning. Work_arrangement and scope minimums are estimates — first V7 labeling run will reveal actual distributions and may require adjustment.

6. ✅ **V7 prompt validation** — Labeled val_unlabeled.jsonl (239 jobs) 3 times, fixing prompt issues each iteration. Run 1: 2 failures (combo ordering). Run 2: 5 failures (untracked tech in array). Run 3: 0 token failures (1 timeout). Prompt locked. Added preflight + input validation + fast-fail guards to label-jobs-v7.ts (Finding 33, 34). All 8 downstream scripts updated for new field names (_raw, short fields, tech arrays).

7. ✅ **Label test/eval set** — Labeled test_unlabeled.jsonl (239 jobs) and val_unlabeled.jsonl (239 jobs) with V7 teacher prompt. Will be locked with chmod 444 after verification.
    - **[2026-03-10 UPDATE]** Model switched from gpt-4o-mini to **gpt-4.1-mini** — better instruction following, 0 fuzzy corrections on test/val sets.
    - `max_completion_tokens` bumped from 500 to 1200 (fixed truncation on long JDs).
    - 0 parse failures across both sets.
    - Script guards added: preflight API check, non-retryable fast-fail (401/403/404), auto-ID for empty job_ids, tech dedup + OOS cleanup.
8. ✅ **Label training data** — Labeled train_unlabeled.jsonl (722 jobs) with V7 teacher prompt (gpt-4.1-mini, temperature=0).
    - **[2026-03-10 UPDATE]** All 1,200 jobs labeled: test=239, val=239, train=722 (3 batches: 300+300+122).
    - 0 parse failures across all data.
    - 7 total fuzzy corrections across all data (5 OOS-mixed-with-real tech tokens, 2 invented comp tokens).
    - Script guards: preflight API check, non-retryable fast-fail (401/403/404), auto-ID for empty job_ids, tech dedup + OOS cleanup.

─── YOU ARE HERE ───────────────────────────────────────────

9. ⬜ **Verify labeling** — Post-label audit. Compare distributions. Spot-check. Combine train batches.
9b. ⬜ **Verify re-labeling** — Post-label audit runs automatically. Also: compare old vs new labels, spot-check 10 random changes. If > 30 labels change, investigate before proceeding. See Step 5.5 checklist.
9. ⬜ **Prune trivially easy bad_fit** — The audit clean mode handles this (`--remove-trivial`). Criteria updated for V7: location=OUTSIDE_UK/UNKNOWN + scope=OUT_OF_SCOPE or seniority=LEVEL_1 + tech=NONE + comp=NO_GBP. Must happen AFTER re-labeling because V7 rules may change some labels.
10. ⬜ **Generate contrastive data** — 140 jobs across 9 batches (A-H, J). Programmatic variants only. Label with V7 teacher at temperature=0. Batch descriptions need V7 token names.
11. ⬜ **Verify contrastive variants** — For each variant: changed field must have different token, unchanged fields must have same tokens. Fix and re-label if not.
12. ⬜ **Run pre-train gates** — Now automated via audit script + build-datasets.ts pipeline. Hard gates: no duplicates, no train/eval overlap, class balance thresholds, source-weight cap, contrastive variant correctness. If ANY fail → STOP and fix.
13. ⬜ **Format and train** — Fresh LoRA from base. 1000 iters. Eval at 600, 700, 800, 900, 1000.
14. ⬜ **Evaluate and decide** — Pick best checkpoint by label accuracy (not val loss). Watch for comp degradation. Compare to V5.1 baseline using score-level mapping (see CLAUDE.md Evaluation Rules).

**Why this order matters**: In V5, labeling happened with incomplete rules, then contrastive data was designed around those incomplete rules. If we'd generated contrastive data first and THEN changed the teacher prompt, we'd have to re-generate and re-label everything. Rules must be locked before any data touches the pipeline. Step 5 (script updates) must happen before Step 6 (re-labeling) because the scripts need to understand V7's 6-field output format.

**Note**: Step 7 changed from "Re-label eval set" to "Build new eval set" due to the data loss incident (see below).

---

## Incident: Accidental Data Loss (2026-03-07)

### What happened
During testing of the newly-wired audit gate in `build-datasets.ts`, the script was run with `--skip-audit` and **default output paths** (which point to production files), then piped through `| head -5`:

```bash
npx tsx src/cli/build-datasets.ts --input data/v5/train_800.jsonl --skip-audit 2>&1 | head -5
```

The script opened `eval_150_golden.jsonl` and `train_800.jsonl` for writing (truncating them to 0 bytes), then `| head -5` killed the process via SIGPIPE before any data was written. Both files zeroed out permanently.

### What was lost
| File | Status | Impact |
|------|--------|--------|
| `data/v5/eval_150_golden.jsonl` | **LOST** (0 bytes, unrecoverable) | The locked V5 eval set (150 jobs, 50/50/50 by label). SHA-256 recorded in manifest but file cannot be recovered — was never in git. |
| `data/v5/train_800.jsonl` | **LOST** (0 bytes) | A cleaned subset of train_860.jsonl. Less critical since train_860.jsonl (the superset) survives. |
| `data/v5/split_manifest.json` | **OVERWRITTEN** | Now contains stale data from a failed rebuild attempt. Marked with `_note: STALE`. |

### What survived
| File | Status |
|------|--------|
| `data/v5/train_860.jsonl` (895 jobs) | INTACT — original V5.1 training data, now chmod 444 |
| `data/v5/all_labeled_pool.jsonl` (1522 jobs) | INTACT — full labeled pool, now chmod 444 |
| `data/v5/mlx/train.jsonl` (806 jobs) | INTACT — MLX chat format |
| `data/v5/mlx/valid.jsonl` (89 jobs) | INTACT — MLX chat format |
| `data/v6/mlx/` (54 train + 6 valid) | INTACT — H/I/J batches |
| All `eval_results/` prediction files | INTACT |
| All adapter checkpoints | INTACT |

### Recovery attempted
1. Tried matching eval predictions file (148 predictions) against all_labeled_pool.jsonl — only 122/148 matched (common titles like "Software Engineer" are ambiguous)
2. Rebuilt from pool with seed=42 — different split (132 eval vs original 150) because pool has grown since original split
3. **Conclusion**: eval_150_golden.jsonl is unrecoverable. V6 must build a fresh eval set.

### Safeguards implemented (4 layers)
1. **Script: overwrite protection** — `build-datasets.ts` now refuses to write to non-empty output files without `--force`. Also refuses if input path = output path (would destroy source). Read-only files shown as "READ-ONLY" in error message.
2. **Script: auto-lock** — `build-datasets.ts` applies `chmod 444` to eval output file immediately after writing.
3. **CLAUDE.md rules** — Added rules 9-11:
   - Rule 9: Never test write-capable scripts with default output paths
   - Rule 10: Never pipe write-capable scripts through `| head` (SIGPIPE kills after truncation)
   - Rule 11: Verify input files exist and are non-empty before running scripts
4. **File permissions** — `chmod 444` applied to `train_860.jsonl` and `all_labeled_pool.jsonl` (surviving critical files).

### Impact on V6 plan
- Step 5 changed from "Re-label eval set" → "Build new V6 eval set from scratch"
- V5/V5.1 eval results remain valid for comparison (prediction files survived) but cannot be re-run on the same eval set
- No impact on training data pipeline (train_860.jsonl intact)

---

## Infrastructure: Script Renames (2026-03-07)

**Why**: All V5-era scripts had version numbers baked into filenames (e.g., `build-v5-datasets.ts`, `label-jobs-v5.ts`). With V6 work underway, this was confusing — "why am I running a v5 script for v6 training?" The scripts are version-agnostic (they take data paths as arguments), so the version numbers added no value and caused confusion.

**Renames performed** (all were untracked files, used `mv` not `git mv`):
| Old name | New name |
|----------|----------|
| `build-v5-datasets.ts` | `build-datasets.ts` |
| `build-v5-input-pool.ts` | `build-input-pool.ts` |
| `label-jobs-v5.ts` | `label-jobs.ts` |
| `check-distribution-v5.ts` | `check-distribution.ts` |
| `format-v5-for-mlx.ts` | `format-for-mlx.ts` |
| `generate-synthetic-v5.ts` | `generate-synthetic.ts` |
| `generate-synthetic-recipes-v5.ts` | `generate-synthetic-recipes.ts` |
| `eval_finetuned_v5.py` | `eval_student.py` |

**Note**: `eval_finetuned_v5.py` → `eval_student.py` (not `eval_finetuned.py` — that already exists for old teacher v1/v2 numeric score eval). The new name better describes its purpose.

**References updated** in 9 files: CLAUDE.md, V6_STUDENT_TRAINING_PLAN.md, V6_DIAGNOSTIC_FINDINGS.md, data/v5/PIPELINE.md, generate-synthetic-hij.ts, generate-synthetic.ts, build-datasets.ts, lora_config_v6.yaml, semantic_tokens.py.

**Zero stale references remaining** — verified with grep across entire codebase.

### Step 7: Final distribution check
Before training, verify:
- good_fit > 12% of total
- maybe > 25% of total
- bad_fit < 60% of total
- ABOVE_100K does NOT exceed RANGE_75_99K in comp distribution
- NO_GBP stays > 55% of comp tokens (matches real world)
- No single contrastive base JD appears more than 3 times
- Zero overlap between training and eval sets (hash check)

### Step 8: Format and train
```yaml
# lora_config_v6.yaml
model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
rank: 16
alpha: 32
dropout: 0.1
learning_rate: 5e-5
batch_size: 1
grad_accumulation: 16  # effective batch = 16, prevents OOM
iters: 1000
warmup: 50
eval_every: 50
save_every: 50
val_batches: 40
mask_prompt: true
grad_checkpoint: true
```

Format all training data with updated student prompt (includes valid token list). Fresh LoRA from base — never continue from V5.1 adapter.

### Step 9: Evaluate
Run eval at checkpoints 600, 700, 800, 900, 1000 on eval_150_golden.jsonl.

Show comparison table:
```
Checkpoint | Label | loc | role | tech | comp | Invalid | Parse
```

Select best checkpoint by **label accuracy**, not val loss (val loss doesn't correlate well with label accuracy for this task).

**Watch for comp degradation** — if comp starts dropping while loc/role climbs, the best checkpoint is just before comp falls. This was the pattern in both V5.0 and V5.1.

### Step 9.5: Ablation Design (if needed)

If V6 results are below target and the cause is unclear, run ablation experiments to isolate which changes helped.

**Key constraint**: Do NOT use 500-iter ablation runs as a screen. V5.1 history shows:
- Best checkpoint arrived late (875 out of 890 before OOM)
- Field accuracy curves are non-monotonic (comp oscillates wildly between checkpoints)
- 500-iter results would have shown 75% accuracy, missing the 84% peak at 875

**Minimum trustworthy comparison**: 700-750 iters as a quick screen. 1000 iters for real decisions.

**Ablation candidates** (run only if V6 misses target):
1. V6 data with V5 teacher labels (isolates prompt fix impact)
2. V6 data without contrastive batches (isolates contrastive training impact)
3. V6 data without pruning (isolates class rebalancing impact)

Each ablation is a full 1000-iter run with eval at 600, 700, 800, 900, 1000. Compare best checkpoint from each against V6 best checkpoint.

### Step 10: Ship or patch
- If ≥ 90%: done
- If 87-89%: implement GPT-4o-mini tiebreaker for boundary scores (45-54, 65-74) → system accuracy 93%+
- If < 87%: analyse new error patterns, generate targeted patch data, retrain once more
- For 96%: upgrade to Qwen2.5-1.5B-Instruct-4bit with same data (one config line change)

---

## Risks and Mitigations

| Risk | Mitigation | Status |
|------|------------|--------|
| OOM crash during training | Batch size 1 + grad_accum 16 (half peak memory vs batch 2) | Planned |
| Comp degrades in later checkpoints | Monitor per-field accuracy at each eval, pick checkpoint before comp drops | Planned |
| Contrastive variants not truly identical | Programmatic string replacement, not GPT rephrasing | Planned |
| GPT-4o-mini labels contrastive variants inconsistently | Temperature=0, verify variants after labeling | Planned |
| Pruning removes too many jobs | Only remove score 0-20 bad_fit. Keep all score 30-49 (near boundary) | Planned |
| Not enough good_fit after pruning | Contrastive batches A, C, E naturally produce good_fit variants. Check at Step 7 | Planned |
| 140 contrastive from ~40 base JDs causes memorisation | Different industry/style/length per base. Max 3 variants. 15% of total dataset | Planned |
| Training data had mixed temperature labels | Step 5: re-label everything at temperature=0 for uniform consistency | Planned |
| V6 re-labeling changes > 30 training labels | Spot-check 10 random changed labels. If V6 rules produce unexpected output, revise prompt before proceeding | Planned |
| "architect" without SENIOR_PLUS misclassifies real senior architects | FALLBACK rule catches them via description. "Chief Architect" fixed — added "chief" to SENIOR_PLUS. Remaining edge: "Associate Architect" correctly falls to FALLBACK → NO_SENIORITY | ✅ Fixed |
| 500-iter ablation gives misleading results | Model peaks late (875 in V5.1). Minimum 700-750 screen, 1000 for real decisions. Never trust early checkpoints | Planned |
| Execution order wrong → wasted work | Follow reordered priorities: lock rules → dedupe → re-label → verify → prune → contrastive → gates → train | ✅ Reordered |
| **GPT-4o-mini reads JD title instead of Title: field** | Pre-label audit flags JD-embedded titles. Post-label title-echo check catches mismatches. Promptfoo tests adherence on known problem jobs. Manual correction of 5 known jobs if needed. (Finding 24) | ✅ Designed |
| **Accidental data loss (SIGPIPE, default paths)** | 4-layer safeguards: overwrite protection (`--force`), auto-chmod 444, CLAUDE.md rules 9-11, file permissions on critical data. (Finding 22) | ✅ Implemented |
| **V5 eval set lost — cannot compare V6 vs V5 on same eval** | Build fresh V6 eval set. V5 prediction files survive for historical reference. Accept that V5↔V6 comparison is approximate, not exact. | Accepted |
| **Data quality issues reach training undetected** | Two-gate validation: pre-label audit (before API calls) + post-label audit (before training). Promptfoo tests prompt against 28 edge cases. Per-run logging for future debugging. Catches all 9 known V5/V5.1 issue types. (Finding 25) | ✅ Designed |
| **New/unknown data quality issues not caught by audits** | Audits only catch known issue types. Mitigated by: spot-checking random samples, monitoring training loss curves for anomalies, comparing field accuracies across checkpoints. Accept residual risk. | Watch |
| **Pre-label JD title detection has false positives/negatives** | Heuristic regex, not perfect. Flags for human review, not auto-reject. May need tuning after first real run. | Accepted |
| **Distribution thresholds may need tuning** | 50% token cap and 15% boundary zone are educated guesses. May trigger false warnings or miss real issues. Adjust based on V6 results. | Watch |

---

## Expected Outcome

```
                V5.1 (875)   V6 (target)
loc:            92.6%        95-97%
role:           92.6%        94-96%
tech:           72.5%        80-85%
comp:           78.5%        85-90%
invalid:        0.7%         <1%
parse fail:     0            0
label accuracy: 83.9%        90-93%

good_fit:       78%          88-92%
maybe:          78%          85-90%
bad_fit:        96%          96-98%
```

The remaining 7-10% errors will be genuine 0.5B-4bit capacity limits. For 93%+ system accuracy without model changes, add the GPT-4o-mini tiebreaker for boundary scores. For 96% model accuracy, upgrade to Qwen2.5-1.5B-Instruct-4bit with the same V6 dataset.
