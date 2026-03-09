# V6 Diagnostic Findings — Pre-Training Analysis

This document records all problems discovered during V5.1 analysis, how they were found, their impact on accuracy, the decisions made, and alternatives considered.

**Important**: Many "decisions" here are hypotheses — we believe V6 fixes them but haven't verified yet. Nothing is confirmed fixed until we re-label and check the verification checklist in V6_STUDENT_TRAINING_PLAN.md Step 5.5. Status column in the summary table distinguishes between "✅ Fixed" (code change made), "Planned" (will be verified after re-labeling), and "**BLOCKER**" (must resolve before re-labeling).

---

## Finding 1: Checkpoint Reference Error (850 vs 875)

**How found**: Cross-referenced CLAUDE.md ("checkpoint 850 = 83.9%") against actual eval result files. Ran both checkpoints and found 850 = 76.7%, 875 = 83.9%.

**Impact**: Without this fix, Phase 1 diagnostics would have run on the wrong checkpoint, producing misleading generalization gap numbers.

**Decision**: Updated all references to 875. The 850 checkpoint still exists for comparison.

**Lesson**: Always verify checkpoint numbers against actual eval output files, not documentation alone.

---

## Finding 2: Temperature Consistency Confirmed

**How found**: Read `src/cli/label-jobs.ts` line 163 — confirmed `temperature: 0`. However, the file is untracked (never committed to git), so there's no version history to verify if earlier labeling runs used a different temperature.

**Impact**: If some jobs were labeled at temperature > 0, their labels could be non-deterministic (same job → different labels on repeated runs). This would create conflicting training signals.

**Decision**: Cannot verify past temperature. The V6 plan includes re-labeling ALL jobs with the new teacher_v6.txt prompt at temperature=0 anyway, so any past inconsistency will be wiped.

**Alternative considered**: Re-label only suspect jobs. Rejected because re-labeling everything costs ~$1 and guarantees uniform consistency.

---

## Finding 3: Generalization Gap Analysis (875 on Training Data)

**How found**: Ran the 875 checkpoint on its own training data (762 jobs from train_800.jsonl).

**Results**:
| Metric | Train Accuracy | Eval Accuracy | Gap |
|--------|---------------|---------------|-----|
| Label | 92.5% | 83.9% | **8.6%** |
| loc | 97.1% | 92.6% | 4.5% |
| role | 95.0% | 92.6% | 2.4% |
| tech | 84.5% | 72.5% | **12.0%** |
| comp | 88.0% | 78.5% | **9.5%** |
| good_fit | ~95% | 78% | **17%** |

**Impact**: The 8.6% overall gap confirms BOTH data quality issues (conflicting labels the model can't learn) AND coverage issues (eval patterns not in training). The tech gap (12%) and comp gap (9.5%) are the biggest contributors.

**Decision**: Address both: clean conflicting labels (data quality) AND add contrastive training data (coverage).

**Alternative considered**: Only add more data without cleaning. Rejected because conflicting labels actively teach the model wrong patterns — more data can't fix that.

---

## Finding 4: Token Frequency Imbalance

**How found**: Counted token frequencies across all 762 training jobs.

**Critical imbalances**:
| Token | Count | % of Field | Problem |
|-------|-------|------------|---------|
| NONE (tech) | 415 | 54.5% | Majority default — model learns "when in doubt, NONE" |
| NO_GBP (comp) | 489 | 64.2% | Same default bias |
| MID_LEVEL (role) | 60 | 7.9% | Severely underrepresented |
| JS_TS_AI_ML (tech) | 7 | 0.9% | Nearly absent — model can't learn this |
| NODE_AI_ML (tech) | 15 | 2.0% | Very rare |

**Impact**: The model defaults to majority tokens when uncertain. This directly explains the top tech errors: AI_ML→NONE (38 errors), JS_TS→NONE (22 errors). The model hasn't seen enough non-NONE tech examples.

**Decision**: Contrastive batches B and D specifically target tech variety. Pruning trivially-easy bad_fit (which are disproportionately NONE + NO_GBP) will also rebalance.

**Lesson**: Token frequency matters more than total job count. 7 examples of JS_TS_AI_ML in 762 jobs means the model sees this combo ~0.9% of the time — not enough to learn.

---

## Finding 5: Boundary Zone Starvation

**How found**: Counted training jobs by score zone and compared against eval error distribution.

**Results**:
| Score Zone | Training Jobs | % | Boundary? |
|-----------|--------------|---|-----------|
| 0-29 | 416 | 54.6% | No — trivially bad_fit |
| 30-49 | 108 | 14.2% | bad_fit↔maybe boundary |
| 50-64 | 138 | 18.1% | Core maybe zone |
| **65-74** | **41** | **5.4%** | **maybe↔good_fit boundary** |
| 75-100 | 59 | 7.7% | Core good_fit zone |

**Impact**: The 65-74 boundary zone has only 41 jobs (5.4%) but accounts for a disproportionate share of eval errors. 11 of the 24 eval errors are good_fit→maybe misclassifications where the score is near this boundary.

**Decision**: Contrastive batches A and H specifically create jobs in the 50-65 zone where comp changes push them across the boundary. This doubles the model's exposure to boundary-region jobs.

**Alternative considered**: Oversample existing boundary jobs. Rejected because duplicating data teaches memorization, not generalization.

---

## Finding 6: Conflicting Labels — "Engineering Manager"

**How found**: Searched training data for jobs with "manager" in title and checked their role tokens.

**Results**: "Engineering Manager" was labeled 3 different ways:
- SENIOR_PLUS: 5 occurrences
- NO_SENIORITY: 4 occurrences
- MID_LEVEL: 1 occurrence

Additionally: "Engineering Manager (React Native)" appeared 4 times (duplicate job ID `4363903342` with `_dup_` suffixes, identical JD text) — all labeled NO_SENIORITY.

**Impact**: These 10 jobs teach the model three contradictory things about the same title. The model oscillates between them across checkpoints, never converging.

**Root cause**: The V5 teacher prompt's keyword list doesn't include "manager". GPT-4o-mini sometimes applies the fallback rule (→ NO_SENIORITY) and sometimes uses semantic understanding (→ SENIOR_PLUS), producing inconsistent labels.

**Decision**: The V6 teacher prompt (teacher_v6.txt) fixes this with an explicit engineering gate + "manager" in SENIOR_PLUS indicators. Re-labeling all jobs with V6 prompt will make all "Engineering Manager" labels consistently SENIOR_PLUS.

**Alternative considered**: Manually fix just the conflicting labels. Rejected because the root cause is the prompt — fixing labels without fixing the prompt means contrastive data will also be inconsistently labeled.

**Lesson**: When the same title gets different labels, the problem is always in the rules, not the model. Fix the rules first.

---

## Finding 7: 21 Unjustified SENIOR_PLUS Labels

**How found**: Extracted all jobs with role=SENIOR_PLUS and checked if the title contains any SENIOR_PLUS keyword from the V5 teacher prompt.

**Result**: 21 jobs have SENIOR_PLUS with no keyword justification in the title. These include:
- "Engineering Manager" variants (manager not in V5 keyword list)
- "Director of Engineering" (director not in V5 keyword list)
- "CTO" / "VP Engineering" (not in V5 keyword list)

**Impact**: These 21 jobs are either:
- **Spec violations**: Teacher broke its own rules (labeled SENIOR_PLUS without a matching keyword)
- **Spec gaps**: The rules were incomplete (should have included manager, director, CTO)

**Decision**: The V6 teacher prompt resolves this by explicitly including manager, director, VP, CTO in SENIOR_PLUS indicators. These titles ARE legitimately senior engineering — the V5 prompt just didn't list them. All 21 will re-label correctly under V6 rules.

---

## Finding 8: Duplicate Jobs Amplifying Bad Labels

**How found**: Searched for duplicate job IDs and identical titles in training data.

**Key duplicates**:
- `4363903342` "Engineering Manager (React Native)": 4 copies, all NO_SENIORITY (wrong under V6 rules → should be SENIOR_PLUS)
- `node_engineer_*` "Node.JS Engineer": 4 copies with synthetic JD variations

**Impact**: Duplicates amplify wrong labels. Four copies of a wrong label teaches the model that wrong answer with 4× weight. The "Engineering Manager (React Native)" duplicates actively teach the model that Engineering Manager = NO_SENIORITY.

**Decision**: Deduplicate (keep 1 copy max per job ID) AND re-label under V6 rules.

**Lesson**: Duplicates are worse than missing data. One wrong duplicate teaches the model more strongly than one correct unique example teaches the right answer.

---

## Finding 9: Checkpoint Stability — 38 Oscillating Eval Jobs

**How found**: Compared predictions across checkpoints 575, 600, 625, ..., 875 on eval set. Classified each job as:
- **Always correct** (across all checkpoints after 575)
- **Oscillating** (flips between correct and incorrect)
- **Always wrong** (never correct at any checkpoint)

**Results**:
- Always correct: 73 jobs (49%)
- Oscillating: 38 jobs (25%)
- Always wrong: 7 jobs (5%)
- Other (partial): 31 jobs (21%)

**Impact**: 38 oscillating jobs (25% of eval) are the most concerning — the model has the capacity to get them right but can't consistently. This indicates the training data contains contradictory signals for these patterns.

**Decision**: Cross-reference oscillating jobs against training data for conflicting labels. The 7 always-wrong jobs are likely capacity limits (accept) or eval label errors (verify).

---

## Finding 10: 77 Non-Software "Engineer" Titles

**How found**: Grep for "engineer" in training job titles, then manually checked which are NOT software/data/platform engineering.

**Examples**: "Sales Engineer", "Support Engineer", "Field Engineer", "Process Engineer", "Mechanical Engineer", "Solutions Engineer", "Customer Engineer".

**Impact**: Under V5 rules, these pass the "engineer" keyword check and get seniority-classified. A "Senior Sales Engineer" gets SENIOR_PLUS when it should get NO_SENIORITY (not a software role).

**Decision**: V6 teacher prompt's engineering gate (Step 1) explicitly defines engineering titles and provides non-engineering examples. "Sales Engineer" → fails gate → NO_SENIORITY regardless of "Senior" in title.

**Alternative considered**: Blocklist of non-engineering titles. Rejected because a positive definition ("what IS engineering") is more robust than a negative one ("what ISN'T") — new titles we haven't seen will be correctly classified by semantic fit rather than missed by an incomplete blocklist.

---

## Finding 11: Teacher Prompt Bug — "architect" in SENIOR_PLUS

**How found**: Reviewed V6 teacher prompt draft for edge cases. Found that "architect" appears in BOTH the engineering gate (line 61) AND the SENIOR_PLUS indicators (line 77).

**Problem**: "Associate Solutions Architect" passes gate 1 (has "architect" → engineering role) then hits SENIOR_PLUS (has "architect" → senior). But "Associate" indicates junior level.

**Decision**: Removed "architect" from SENIOR_PLUS indicators. Architects still pass the engineering gate but now fall through to the FALLBACK rule, where the description determines seniority. An "Associate Solutions Architect" will correctly get NO_SENIORITY (no senior keywords in description), while a "Principal Architect" gets SENIOR_PLUS via "principal" keyword.

---

## Finding 12: Teacher Prompt Bug — "leader" Missing from SENIOR_PLUS

**How found**: Batch J item 12 ("Engineering Team Leader") revealed that "Leader" ≠ "Lead" — if only "lead" is in the keyword list, "leader" won't match as a substring in case-insensitive comparison (lead≠leader).

**Decision**: Added "leader" to SENIOR_PLUS indicators. Now both "Team Lead" and "Team Leader" correctly trigger SENIOR_PLUS.

---

## Finding 13: 123 Trivially Easy bad_fit Jobs

**How found**: Counted training jobs matching ALL of: loc=OUTSIDE_UK or MISSING, role=NO_SENIORITY, tech=NONE, comp=NO_GBP, label=bad_fit.

**Result**: 123 jobs (16% of training set).

**Impact**: The model already gets bad_fit 96% correct. These 123 jobs teach nothing new — they just dilute the signal from harder boundary cases. They also skew the class balance to 63% bad_fit.

**Decision**: Remove ~100-120 of these (keep any with score 30-49 near the maybe boundary). Expected rebalancing: bad_fit drops from 63% to ~56%, giving more relative weight to maybe (29%) and good_fit (15%).

---

## Finding 14: Teacher Prompt Design — Positive vs Negative Definitions

**How found**: User insight during prompt design. The V5 teacher prompt listed non-engineering examples ("NOT: Marketing Manager, Product Manager...") as a blocklist. User asked: "why not just give it the job title we are interested in like software engineer?"

**Problem with V5 approach**: A blocklist can never be complete. New non-engineering titles we haven't seen will slip through because they're not on the list. Meanwhile, the model gets confused by trying to memorize what NOT to do.

**Decision**: V6 teacher prompt uses a positive definition — explicitly lists what engineering titles look like (engineer, developer, programmer, architect, SRE, DevOps, etc.) and provides non-engineering examples only for illustration. The gate question is "IS this engineering?" not "is this NOT marketing/sales/HR/...?"

**Why this is better**: A positive definition covers unseen titles by semantic similarity. "Cloud Infrastructure Engineer" passes the gate because it looks like the positive examples. A blocklist would need to have explicitly excluded every non-engineering title.

**Lesson**: In classification rules, always define what the target IS, not what it ISN'T. Positive definitions generalise to unseen examples; blocklists don't.

---

## Finding 15: "→ ignored" Reasoning Format Causes Drift

**How found**: During prompt critique. The V5 and V6 prompts require the model to list ignored technologies in reasoning (e.g., "react → ignored, python → ignored"). This format inflates the reasoning section and may cause the 0.5B student model to lose track of what actually matters vs what's ignored.

**Impact**: Not yet quantified. The teacher (GPT-4o-mini) handles this fine, but the student model has limited context attention. Long reasoning with many "→ ignored" items may dilute the signal.

**Decision**: Keep the "→ ignored" requirement in the teacher prompt (it forces explicit reasoning and prevents silent omissions). Monitor whether the student model's tech errors correlate with reasoning length. If they do, consider simplifying the student prompt's reasoning format in a future version.

**Alternative considered**: Remove "→ ignored" entirely. Rejected because without it, the teacher sometimes silently ignores tech it shouldn't and there's no audit trail.

---

## Finding 16: Teacher Prompt Length

**How found**: During prompt critique. The V6 teacher prompt is ~180 lines. Analysis suggested it could be cut ~40% without losing any rules.

**Decision**: Keep current length for now. The teacher prompt is used with GPT-4o-mini which has 128k context — length is not a constraint. Clarity and completeness are more important than brevity for the teacher. The student prompt (separate file) is already minimal.

**Alternative considered**: Aggressively compress to reduce API cost. Rejected because GPT-4o-mini is cheap (~$0.15/1M input tokens) and the prompt is used ~900 times total (~$0.02). Not worth the risk of introducing ambiguity.

---

## Finding 17: AI/ML Trigger Keywords Not Specified

**How found**: During gap check of teacher_v6.txt. The TECH section defined explicit trigger words for Node (`node.js, nodejs, node`) and JS/TS (`javascript, typescript`) but AI_ML had no keyword list — just "ONLY if explicitly REQUIRED experience."

**Problem**: Without explicit keywords, GPT-4o-mini decides on its own what counts as "AI/ML." This could be inconsistent: does "data pipeline" count? Does "AI-powered" in a company description? Does "NLP" count?

**Decision**: Added explicit AI_ML trigger keyword list: `machine learning, ML, deep learning, LLM, LLMs, NLP, artificial intelligence, AI/ML, computer vision, neural network`. The "ONLY count if explicitly REQUIRED" rule still applies on top.

**Why this matters**: Every other tech category (Node, JS/TS) has explicit triggers. AI_ML without them is a consistency hole — exactly the kind of ambiguity that causes conflicting labels.

---

## Finding 18: "Chief Architect" Edge Case

**How found**: After removing "architect" from SENIOR_PLUS indicators (Finding 11), realized "Chief Architect" would fall through to FALLBACK. "Chief" was not in the SENIOR_PLUS keyword list.

**Decision**: Added "chief" to SENIOR_PLUS indicators. Now "Chief Architect" → passes engineering gate (architect) → SENIOR_PLUS (chief). Without this, it would fall to FALLBACK and likely get NO_SENIORITY.

---

## Finding 19: Title-Data Mismatch — Stored Title ≠ Labeled Title (CRITICAL)

**How found**: During Step 5.2 (spec violation analysis), checked the reasoning field for SENIOR_PLUS jobs without V5 keyword justification. The reasoning reveals the title the teacher actually saw at labeling time. In 7 cases, this is DIFFERENT from the title stored in train_800.jsonl.

**Examples**:
- Stored: "Node.JS Engineer" → Teacher saw: "Senior Backend Developer" → SENIOR_PLUS (correct for teacher's title, WRONG for stored title)
- Stored: "Software Developer (remote)" → Teacher saw: "Senior Full-Stack Engineer"
- Stored: "Machine Learning Engineer (all)" → Teacher saw: "Senior Machine Learning Engineer"
- Stored: "DevOps Engineer" → Teacher saw: "Senior..."

**Impact**: The student model sees `Title: Node.JS Engineer` paired with `role: SENIOR_PLUS`. This teaches the model that plain engineer titles = SENIOR_PLUS. 4 of the 7 are "Node.JS Engineer" duplicates, creating 4× wrong gradient. This directly explains the 25 SENIOR_PLUS→NO_SENIORITY training errors found in Step 2 — the model learned that "Engineer" alone means senior.

**Root cause**: Unknown. Either:
1. Titles were modified/cleaned after labeling (post-processing bug)
2. The labeling script used a different title field than what was stored
3. The title was extracted from the JD text rather than the title field

**Decision**: Must investigate root cause before re-labeling. If the labeling script uses a different field, re-labeling will reproduce the same bug. Check `src/cli/label-jobs.ts` to verify which title field is passed to the teacher prompt.

**Action items**:
1. Trace the labeling pipeline to find where titles diverge
2. Fix the pipeline so the teacher sees the same title the student will see
3. Re-label AFTER the pipeline is fixed

**Lesson**: Always verify that the data the teacher sees at labeling time is IDENTICAL to what the student sees at training time. A mismatch between labeling input and training input creates systematically wrong labels.

---

## Finding 20: Non-Engineering Titles in Spec Gap Category

**How found**: During Step 5.2 analysis. Two jobs classified as "spec gap" (V6 keyword matches) are actually NOT software engineering roles:
- "Regional Director" (line 632) — not engineering
- "Director Fire Engineering" (line 145) — fire engineering, not software

**Impact**: Under V6 rules, these should fail the engineering gate → NO_SENIORITY. The current V5 label (SENIOR_PLUS) is wrong under both V5 and V6 rules. These are actually spec violations, not spec gaps.

**Decision**: V6 re-labeling will fix these automatically (engineering gate rejects them). No manual action needed, but worth tracking to verify.

---

## Finding 21: No Unified Data Quality Tooling

**How found**: When starting Step 5.3 (deduplication), investigated existing cleanup capabilities. Found 8 scripts with partial dedup/validation logic scattered across the codebase, each silently dropping bad data with no audit trail.

**Problem**: No single tool could answer "what's wrong with this training set?" Every script applied its own ad-hoc checks, silently removed jobs, and left no record of what was removed or why. This meant:
- No way to know what the training set actually contained after processing
- No way to catch eval contamination (synthetic jobs sharing title+company with eval set)
- No way to distinguish intentional contrastive pairs from true duplicates
- No quarantine — dropped data was gone forever

**Decision**: Built `src/cli/audit-training-data.ts` — a single-pass audit with 6 critical checks and 7 warning checks. Key design choices:
- **Critical vs warning separation**: Critical issues (eval contamination, invalid tokens) block the pipeline. Warnings (duplicates, suspicious tokens) are reported + quarantined but don't block.
- **Contrastive pair awareness**: Jobs sharing `source_job_id` are detected as intentional variants and protected from dedup.
- **Clean mode**: Produces a cleaned file + quarantine directory, with separate tracking for each removal reason (not lumped together).
- **Pipeline integration**: Wired into `build-datasets.ts` as a pre-flight gate that runs automatically before every dataset build.

**Results on train_860.jsonl**: 17 critical (eval contamination), 107 duplicates, 123 trivially easy bad_fit, 19 contrastive pairs protected. Clean output: 526 jobs from 762.

**Lesson**: Data quality tooling should be built BEFORE you need it, not after you discover problems in training results. If we'd had this audit from V5.0, we would have caught the duplicate "Engineering Manager" amplification (Finding 8) and eval contamination before they affected training.

---

## Finding 22: Accidental Data Loss — SIGPIPE Destroyed Eval Set

**How found**: During testing of the newly-wired audit gate in `build-datasets.ts`. Ran the script with `--skip-audit` and default output paths, piped through `| head -5`.

**What happened**: `build-datasets.ts` opens output files with `fs.createWriteStream()` which truncates immediately. The `| head -5` in the shell pipe sent SIGPIPE after reading 5 lines of stdout, killing the process. Files were already truncated to 0 bytes but no data had been written yet.

**Files lost**:
- `data/v5/eval_150_golden.jsonl` — the locked V5 eval set (150 jobs). **UNRECOVERABLE** — never tracked in git, SHA-256 recorded but content gone.
- `data/v5/train_800.jsonl` — a cleaned subset. Less critical (train_860.jsonl is the superset).

**Root causes** (multiple failures compounded):
1. **Default output paths point to production files** — running without explicit `--eval-output` / `--train-output` targets the real eval set
2. **No overwrite protection** — script happily truncated non-empty files
3. **SIGPIPE kills after truncation** — `| head` is a common testing pattern but lethal for write-capable scripts
4. **Critical data not in git** — no recovery possible from version control
5. **No file permissions** — eval set was writeable despite being "locked"

**Decision**: Implemented 4 layers of defense:
1. Script: `--force` required to overwrite non-empty files + input≠output hard stop
2. Script: auto `chmod 444` on eval files after writing
3. CLAUDE.md: rules against testing with default paths and piping through `| head`
4. File system: `chmod 444` on surviving critical files (train_860.jsonl, all_labeled_pool.jsonl)

**Impact on V6**: Must build a fresh eval set from all_labeled_pool.jsonl. V5 eval results remain valid for historical comparison (prediction files survived) but the eval set itself cannot be re-run.

**Lesson**: Defense in depth for data files — no single safeguard is enough. The eval set had ONE protection ("locked — never modify" in CLAUDE.md), which was a behavioral rule with no enforcement. Four compounding failures (default paths + no overwrite check + SIGPIPE + no git + no chmod) all had to fail for data loss to occur. Each safeguard layer catches a different failure mode.

---

## Finding 23: Script Version Numbers Cause Confusion

**How found**: While working on V6 audit tooling, the user noted: "confusing to have `build-v5-datasets.ts` for the model we are training that is v6."

**Problem**: All V5-era scripts had version numbers in filenames (`build-v5-datasets.ts`, `label-jobs-v5.ts`, `format-v5-for-mlx.ts`, etc.). The scripts are actually version-agnostic — they take data paths as arguments and work for any version. The v5 in the name was misleading and created friction when using them for V6 work.

**Decision**: Renamed all 8 scripts to drop version numbers (e.g., `build-v5-datasets.ts` → `build-datasets.ts`). Updated references in 9 files. Special case: `eval_finetuned_v5.py` → `eval_student.py` (not `eval_finetuned.py` which already exists for the old teacher model).

**Verification**: Grep confirmed zero stale references across entire codebase.

**Lesson**: Don't bake version numbers into tool names unless the tool is genuinely version-specific. Version-agnostic tools with versioned names create confusion and cognitive overhead when the project moves to the next version.

---

## Finding 24: Title-Mismatch Root Cause — GPT-4o-mini Prompt Adherence Failure + LinkedIn Data Quirk

**How found**: Deep investigation of Finding 19. Traced the full labeling pipeline (10 scripts), examined all 895 jobs in train_860.jsonl, and compared stored titles against teacher reasoning.

**Initial hypothesis**: Data manipulation scripts might have modified titles during preprocessing, curation, or augmentation. The project has 8+ scripts that manipulate job data.

**Investigation**: Traced every script in the pipeline — `preprocess-raw-jobs.ts`, `assemble-student-training.ts`, `build-input-pool.ts`, `label-jobs.ts`, `extract-title-salary-description.ts`, `augment-training-data.ts`, `build-datasets.ts`, `format-for-mlx.ts`. **None of them modify the title field.** The most they do is `normalizeText()` which only collapses whitespace. `label-jobs.ts` line 147 passes `job.title` directly into the prompt template — no field swapping, no alternate sources.

**Most likely root cause (strong hypothesis, not fully proven — no labeling logs exist from original runs):**

1. **LinkedIn titles ≠ JD body titles**: Companies post with a generic listing title ("Node.JS Engineer") but their JD text describes a different role ("Senior Backend Developer"). This is a real-world data quirk — the LinkedIn scraper picks up the listing title, but the company wrote their JD with their internal title.

2. **GPT-4o-mini ignores "check title only" instruction**: The V5 teacher prompt (line 14) says `ROLE — check title only (case-insensitive)`. But GPT-4o-mini receives the full JD text via `{{jd_text}}`, finds a more prominent title inside it, and uses THAT instead. The reasoning field proves this — it says `'Senior Backend Developer' in title` when the Title: field said "Node.JS Engineer".

**The 8 affected jobs (5 poisonous, 3 harmless):**

| Stored Title | Teacher Used (from JD body) | Role Given | Correct Role | Poisonous? |
|---|---|---|---|---|
| Node.JS Engineer (×4 duplicates) | Senior Backend Developer | SENIOR_PLUS | NO_SENIORITY/MID_LEVEL | **YES** — 4× wrong gradient |
| Software Developer (remote) | Senior Full-Stack Engineer | SENIOR_PLUS | NO_SENIORITY/MID_LEVEL | **YES** |
| Machine Learning Engineer (all) | Senior Machine Learning Engineer | SENIOR_PLUS | NO_SENIORITY/MID_LEVEL | **YES** |
| DevOps Engineer | Senior DevOps Engineer | SENIOR_PLUS | NO_SENIORITY/MID_LEVEL | **YES** |
| Data Scientist | Data Scientist III | SENIOR_PLUS | NO_SENIORITY/MID_LEVEL | **YES** |
| Perm Tech Leader | Perm Leader | NO_SENIORITY | NO_SENIORITY | No — same result |
| Head Of Of Engineering | Head Of Engineering | SENIOR_PLUS | SENIOR_PLUS | No — typo correction |
| Quality Assurance Tester | QA Tester | NO_SENIORITY | NO_SENIORITY | No — abbreviation |

**Training impact**: The 5 poisonous jobs (8 with duplicates) teach the student that plain "Engineer"/"Developer"/"Scientist" titles = SENIOR_PLUS. This directly explains the 25 SENIOR_PLUS→NO_SENIORITY training errors found in Finding 3. The 4× Node.JS Engineer duplicates amplify this wrong signal with 4× gradient weight.

**Solutions considered:**

| Option | Description | Rejected because |
|---|---|---|
| Instructor (retry-based validation) | Wrap GPT calls with Zod schema, retry on invalid | Designed for capable models. A 0.5B student can't self-correct. Wrong tool. |
| Constrained decoding (Outlines) | Force valid tokens at logits level | Correct tool for student model, but V5.1 already at 0.7% invalid — not worth the MLX integration complexity now |
| Strip JD-embedded titles before labeling | Regex out "Job Title: X", "Role: X" patterns from JD | Adds preprocessing complexity, risk of removing legitimate JD content, extra code that can have bugs |
| Stronger prompt warning in V6 | Add explicit "WARNING: ignore titles in description" | Cheap but may not work — GPT-4o-mini already ignores "check title only" |
| Fix labels manually (5 jobs) | Correct the 5 known jobs by hand | Fixes known jobs but doesn't prevent NEW mismatches in future labeling runs |
| Fix titles to match JD | Change stored title to match JD body | Wrong — at inference time we get the LinkedIn title, not the JD title |
| Separate validation script | New validate-labels.ts | Adds another tool to maintain, scatters validation logic |

**Decision: Pre-label + post-label audit approach**

Instead of any single fix, we designed a two-gate validation system around the labeling step:

```
Input pool → PRE-LABEL AUDIT → Label clean jobs → POST-LABEL AUDIT → Build datasets → Train
               (instant)          (~12 min)          (instant)         (instant)       (hours)
```

**Pre-label audit** (`audit-training-data.ts --pre-label`) catches issues in input data BEFORE spending API money:
- Duplicates (don't label the same job twice — saves ~$0.12)
- Eval contamination (don't label eval jobs)
- JD-embedded title mismatch (flag jobs where JD body contains a different title than the Title: field — the exact Finding 19 problem)
- Short/corrupt JDs
- Trivially easy bad_fit (why label jobs we'll prune anyway?)

**Post-label audit** (existing `audit-training-data.ts` with 3 new checks) catches issues that only appear after GPT responds:
- Title-echo: does the reasoning reference the stored title or a different one?
- Distribution balance: warn if any token exceeds 50% of its field, or boundary zone (score 50-74) has < 15% of jobs
- Label consistency: same title should always get the same role token

**Additional: Promptfoo test config** for V6 prompt — test against the 8 known title-mismatch jobs and other edge cases before any labeling run. Quick sanity check (5 seconds).

**Additional: Per-run log file** in `label-jobs.ts` — timestamped JSONL log of every labeling request/response for future debugging.

**Why this approach:**
- **No new scripts** — extends the existing audit and labeling scripts
- **Validate at the earliest possible point** — every check that can run on unlabeled data runs BEFORE labeling. Checks needing GPT output run AFTER.
- **Never spend money producing data we'll throw away** — pre-label audit removes known-bad jobs before API calls
- **Never spend hours training on data with known problems** — post-label audit is the final gate
- **~65 lines of new code** across 2 existing files + 1 Promptfoo YAML config — minimal complexity
- **Catches ALL 9 known V5/V5.1 data issues** — verified against every finding

**Finding 19 status change**: No longer a BLOCKER for re-labeling. The pre-label audit flags title mismatches before labeling, and the post-label audit verifies GPT respected the title. If GPT still gets it wrong on specific jobs, we manually correct those (known, finite list of 5 jobs). Step 5.5D remains as a verification gate.

**Limitations of this approach:**

1. **Pre-label JD title detection is heuristic** — uses regex/string matching to detect if the JD body contains a different title than the Title: field. Will have false positives (JD mentions a related role that isn't the actual title) and false negatives (title embedded in unusual format we didn't pattern-match). Not a perfect filter — a flag for human review, not an auto-reject.

2. **Post-label title-echo depends on reasoning format** — parses GPT's reasoning string to extract what title it referenced. If GPT changes its reasoning format (e.g., omits the title, uses a different phrasing), the check breaks silently. Tied to the specific `role: '...' in title →` format.

3. **Distribution thresholds are educated guesses** — "warn if any token > 50% of its field" and "boundary zone < 15%" are based on V5/V5.1 experience, not proven thresholds. May trigger false warnings on legitimately skewed data (e.g., NO_GBP genuinely is ~60% in real-world jobs) or miss a real imbalance just under the threshold.

4. **Only catches KNOWN issue types** — the two-gate system covers all 9 V5/V5.1 issues because we designed checks for each. If V6 introduces a NEW type of data quality problem we haven't seen before, neither gate will catch it. We're protected against the past, not the future.

5. **Manual correction is a patch, not a systemic fix** — if GPT still misreads 5 titles after V6 re-labeling, we manually correct them. But if new data arrives with the same LinkedIn title ≠ JD title quirk, we'll need to catch and fix those too. The pre-label audit flags them, but someone still has to decide the correct label.

6. **No guarantee V6 prompt fixes the adherence issue** — the V6 prompt is more structured, but GPT-4o-mini may still read JD body titles. The validation catches this after the fact but doesn't prevent it. We're detecting the problem, not eliminating the root cause.

7. **Promptfoo tests known edge cases only** — 15-20 test cases can't cover 900 real jobs. A new type of GPT misbehavior on a job we didn't test won't be caught by Promptfoo. The post-label audit is the real safety net, not Promptfoo.

8. **Log files grow without cleanup** — no rotation or archival strategy. After many labeling runs, the `labeling_runs/` directory will accumulate large files. Minor issue for now (~900 jobs per run, ~1MB per log), but worth noting.

**Lesson**: When the same data issue keeps causing problems, the answer isn't fixing individual instances — it's building validation into the pipeline so the issue CAN'T reach training. Fix the process, not the data.

---

## Finding 25: V5/V5.1 Had No Data Validation Between Labeling and Training

**How found**: While designing the solution for Finding 24, mapped out where each V5/V5.1 issue would have been caught if validation existed. Realized NONE of them were caught before training — they were all discovered AFTER training by analysing poor eval results.

**The V5/V5.1 data issues and when they were discovered:**
| Issue | When discovered | Hours wasted |
|---|---|---|
| Conflicting "Engineering Manager" labels | After V5.1 training, during error analysis | ~4h training + eval |
| 4× duplicate amplification | After V5.1 training, during error analysis | Same run |
| Title mismatches (8 jobs) | After V5.1 training, during V6 pre-training analysis | Both V5.0 and V5.1 runs |
| 77 non-engineering titles | After V5.1 training, during V6 analysis | Both runs |
| 17 eval-contaminated jobs | During V6 audit tooling build | Both runs (inflated eval scores) |
| 123 trivially easy bad_fit | During V6 analysis | Both runs (wasted training capacity) |
| Token frequency imbalance | During V6 analysis | Both runs |
| Boundary zone starvation | During V6 analysis | Both runs |

**Impact**: Two full training runs (V5.0 and V5.1, ~4h each) produced results below target, and the root causes were all data quality issues that could have been detected BEFORE training with automated checks.

**What the V6 validation pipeline would have caught — and when:**

Most of these issues trace back to **prompt problems**, not just data problems. A Promptfoo test config with edge case assertions would have caught the prompt issues before a single job was labeled. The pre/post-label audits would have caught the data issues before training.

| V5/V5.1 Issue | Root cause | What catches it in V6 | When |
|---|---|---|---|
| Conflicting "Engineering Manager" labels | V5 prompt missing "manager" keyword | **Promptfoo**: test "Engineering Manager" → SENIOR_PLUS | Before labeling (5 sec) |
| 14 spec gap titles (Director, CTO, Sr.) | V5 prompt incomplete keyword list | **Promptfoo**: test each title → SENIOR_PLUS | Before labeling (5 sec) |
| "architect" over-matching in SENIOR_PLUS | V5 prompt had architect in wrong list | **Promptfoo**: test "Associate Solutions Architect" → NOT SENIOR_PLUS | Before labeling (5 sec) |
| "leader" missing from SENIOR_PLUS | V5 prompt incomplete keyword list | **Promptfoo**: test "Engineering Team Leader" → SENIOR_PLUS | Before labeling (5 sec) |
| No AI/ML keyword list → inconsistent labels | V5 prompt left AI/ML undefined | **Promptfoo**: test required vs mentioned vs nice-to-have | Before labeling (5 sec) |
| 77 non-engineering titles getting seniority | V5 prompt had no engineering gate | **Promptfoo**: test "Sales Engineer", "Regional Director" → NO_SENIORITY | Before labeling (5 sec) |
| 8 title mismatches (GPT reads JD title) | GPT ignores "check title only" + LinkedIn data quirk | **Promptfoo**: test Node.JS Engineer with Senior JD body. **Pre-label audit**: flag JD-embedded titles. **Post-label audit**: title-echo check | Before labeling + after labeling |
| Incomplete COMP ignore rules | V5 prompt missing TC/OTE/daily/package rules | **Promptfoo**: test "TC £120k", "£500/day", "OTE £110k" → NO_GBP | Before labeling (5 sec) |
| 4× duplicate amplification | No dedup in pipeline | **Pre-label audit**: duplicate detection | Before labeling (instant) |
| 17 eval-contaminated jobs | No contamination check | **Pre-label audit**: eval contamination | Before labeling (instant) |
| 123 trivially easy bad_fit | No pruning strategy | **Pre-label audit**: trivial bad_fit detection | Before labeling (instant) |
| Token frequency imbalance (54% NONE) | No distribution monitoring | **Post-label audit**: distribution balance check | After labeling (instant) |
| Boundary zone starvation (5.4%) | No zone monitoring | **Post-label audit**: distribution balance check | After labeling (instant) |

**Key insight: 8 of the 13 issues are prompt problems that Promptfoo catches in 5 seconds.** If we'd had a Promptfoo config testing edge cases before V5 labeling, we would have discovered that "manager" was missing, "architect" was over-matching, the engineering gate didn't exist, and COMP ignore rules were incomplete — all before labeling a single job. The remaining 5 issues are data quality problems caught by the pre/post-label audits.

The total cost of NOT having this validation: two training runs (~8h GPU time), two eval cycles, and weeks of diagnostic analysis to find root causes that a 5-second Promptfoo run + instant audit would have caught upfront.

**Decision**: The two-gate validation system (Finding 24) ensures V6 doesn't repeat this pattern. Every known issue type has a corresponding automated check that runs before training begins. The Promptfoo config is the highest-ROI component — it tests the prompt itself (the source of most issues) in seconds.

**Lesson**: The most expensive place to discover a data quality issue is after training. The cheapest place is before labeling. But even cheaper than that: **test the prompt against known edge cases before labeling anything.** Most data quality issues in V5/V5.1 were actually prompt quality issues — wrong rules producing wrong labels at scale. A 5-second prompt test would have saved 8+ hours of training and weeks of analysis.

---

## Summary of Decisions

**Status key**: ✅ Code change made | ⏳ Hypothesis — needs verification after re-labeling | Planned — not started | **BLOCKER** — must resolve before re-labeling

| # | Finding | Decision | Status |
|---|---------|----------|--------|
| 1 | Checkpoint reference error | Fixed CLAUDE.md (850→875) | ✅ Done |
| 2 | Temperature confirmed = 0 | Re-label everything with V6 prompt anyway | Planned |
| 3 | Generalization gap 8.6% | Fix data quality + add contrastive coverage | Planned |
| 4 | Token frequency imbalance | Pruning + contrastive batches B, D | Planned |
| 5 | Boundary zone starvation | Contrastive batches A, H target boundary | Planned |
| 6 | "Engineering Manager" conflicting labels | V6 prompt adds "manager" to SENIOR_PLUS | ⏳ Verify after re-label |
| 7 | 24 unjustified SENIOR_PLUS (was 21) | 14 spec gaps, 7 title mismatches, 2 violations, 1 V6-wrong | ⏳ Verify after re-label |
| 8 | Duplicate job amplification | Deduplicate before re-labeling | Planned |
| 9 | 38 oscillating eval jobs | Cross-reference with training conflicts | Planned |
| 10 | 77 non-software "engineer" titles | V6 engineering gate (semantic, not keyword) | ⏳ Verify after re-label |
| 11 | "architect" bug in SENIOR_PLUS | Removed from SENIOR_PLUS, kept in gate | ✅ Code change |
| 12 | "leader" missing from SENIOR_PLUS | Added to SENIOR_PLUS indicators | ✅ Code change |
| 13 | 123 trivially easy bad_fit | Prune ~100-120, keep near-boundary | Planned |
| 14 | Positive vs negative definitions | V6 uses positive engineering definition | ✅ Code change |
| 15 | "→ ignored" reasoning drift | Keep in teacher, monitor student impact | Watch |
| 16 | Teacher prompt length (~180 lines) | Keep — GPT-4o-mini cost is negligible | No action |
| 17 | AI/ML trigger keywords not specified | Added explicit keyword list to teacher_v6.txt | ✅ Code change |
| 18 | "Chief Architect" edge case | Added "chief" to SENIOR_PLUS indicators | ✅ Code change |
| 19 | Title-data mismatch (8 jobs, 5 poisonous) | Pre-label audit flags, post-label audit verifies, manual fix if needed | ⏳ Verify after re-label |
| 20 | Non-engineering titles in spec gaps (2 jobs) | V6 gate should fix — uncertain | ⏳ Verify after re-label |
| 21 | No unified data quality tooling | Built audit-training-data.ts with 13 checks, pipeline gate, clean mode | ✅ Done |
| 22 | SIGPIPE destroyed eval set + train file | 4-layer safeguards: overwrite protection, auto-chmod, CLAUDE.md rules, file permissions | ✅ Done |
| 23 | Script version numbers cause confusion | Renamed 8 scripts to drop v5 suffix, updated 9 files | ✅ Done |
| 24 | Title-mismatch root cause: GPT adherence + LinkedIn quirk | Two-gate validation: pre-label audit + post-label audit + Promptfoo + logging. Root cause is strong hypothesis (pipeline traced, plausible), not fully proven (no logs from original runs). | ✅ Implemented (pre-label, post-label, Promptfoo, logging all built) |
| 25 | No data validation between labeling and training in V5/V5.1 | Two-gate system catches all 9 known issue types before training | ✅ Implemented |
| 26 | NODE token family severely underrepresented | 2,543 real + 46 tech-swap synthetic jobs. All NODE gaps filled. | ✅ Done |
| 27 | Structured reasoning format redesign | 8-field interleaved format replaces free-form reasoning. 3-layer defense: teacher prompt, audit check, normalization. | ✅ Implemented |
| 28 | V7 teacher prompt architecture (4→6 fields) | 6 fields, 28 tokens, richer vocabulary. Backward-compatible scoring via translation layer. All downstream scripts updated. | ✅ Implemented |
| 29 | V6→V7 distribution gaps (FULLY_REMOTE, NODE, NODE_AI_ML) | 60 FULLY_REMOTE + 45 NODE + 10 NODE_AI_ML synthetic variants from real JDs. V7 minimums added to plan. | ✅ Done |

---

## Lessons Learned

1. **Fix the rules before fixing the data**: Conflicting labels always trace back to ambiguous rules. Fixing individual labels without fixing the prompt is a patch that will break on new data.

2. **Positive definitions > negative definitions**: "What IS engineering" is more robust than "what ISN'T" — you can't enumerate every non-engineering title.

3. **Duplicates are actively harmful**: One wrong duplicate is worse than missing data. Four copies of a wrong label creates 4× gradient in the wrong direction.

4. **Boundary zones need proportional representation**: 5.4% of training at the zone responsible for ~50% of errors guarantees poor performance.

5. **Token frequency drives default behavior**: A 0.5B model with 54.5% NONE tech tokens will default to NONE whenever uncertain — it's the statistically safe bet.

6. **Checkpoint verification is non-negotiable**: Documentation can drift from reality. Always verify against actual output files.

7. **Keyword lists are fragile**: "lead" ≠ "leader", "manager" omitted, "architect" over-matches. Semantic framing (engineering gate) is more maintainable than exhaustive keyword enumeration.

8. **Don't trust early checkpoints for ablation**: V5.1 peaked at 875 out of 890 iters. Field accuracy curves are non-monotonic (comp oscillates). A 500-iter ablation would show 75% and miss the 84% peak. Minimum screen: 700-750 iters. Real decisions: 1000 iters.

9. **Execution order matters**: Lock rules → dedupe → re-label → verify → prune → contrastive → gates → train. Doing any step out of order risks wasted work (e.g., generating contrastive data before rules are locked means re-labeling contrastive data too).

10. **A prompt change is a hypothesis, not a fix**: Writing V6 prompt rules doesn't mean they work. The engineering gate is semantic — GPT-4o-mini might still let "Sales Engineer" through. Every claim about V6 fixing something must be verified with actual re-labeled data. "Fixed in prompt" ≠ "Fixed in data."

11. **Verify the full pipeline, not just the prompt**: Even a perfect prompt produces wrong labels if the pipeline feeds it wrong inputs. The title mismatch (Finding 19) proves this — the teacher prompt was fine, but the wrong title was passed to it.

12. **Build data quality tooling before you need it**: If we'd had the audit script from V5.0, we would have caught duplicate amplification and eval contamination before they affected training results. Retrofitting quality checks after training errors is more expensive than building them upfront.

13. **Defense in depth for critical data**: No single safeguard is reliable. The V5 eval set had ONE protection (a CLAUDE.md rule saying "locked — never modify"), which had no enforcement. Four compounding failures (default paths + no overwrite check + SIGPIPE + no git + no chmod) all had to fail simultaneously. Each safeguard layer catches a different failure mode — behavioral rules catch human mistakes, script guards catch programmatic mistakes, file permissions catch OS-level mistakes.

14. **SIGPIPE is lethal for write-capable scripts**: `| head -N` is a common testing pattern that sends SIGPIPE after N lines. If the script opens output files before writing stdout, files get truncated to 0 bytes and the process dies before writing any data. Never pipe write-capable scripts through head/tail — redirect to a file instead.

15. **Don't bake version numbers into tool names**: Version-agnostic tools with versioned names (`build-v5-datasets.ts`) create confusion when the project moves to the next version. Name tools by what they do, not when they were written.

16. **"Check title only" doesn't mean the model will**: GPT-4o-mini receives the full JD alongside the title field. When the JD contains a more prominent title (e.g., "Senior Backend Developer"), the model uses it despite explicit instructions to use only the Title: field. Prompt adherence is not guaranteed — validate the output, don't trust the instruction alone.

17. **Validate at the earliest possible point**: The most expensive place to discover a data quality issue is after training (~4h). The cheapest is before labeling (~0s). Split validation into two gates: pre-label (catches input data issues before spending API money) and post-label (catches model output issues before spending training hours). Every check that can run on raw input should run before labeling. Never spend money producing data you'll throw away, never spend hours training on data with known problems.

18. **When investigating data issues, check the full pipeline before concluding**: Initial hypothesis for Finding 19 was "GPT doesn't follow instructions." User pushed back: "we have scripts that manipulate data." Investigation of 10 scripts confirmed the pipeline was clean — but we would have been wrong to skip the check. Always trace the full data flow, not just the last step.

19. **For tiny models, reasoning quality matters as much as label quality**: Free-form reasoning teaches stylistic variation that wastes model capacity. Structured reasoning with predictable patterns lets the model focus capacity on classification accuracy. A 0.5B model can't afford to spend parameters learning that "→ UK" and "→ UK_OTHER" mean the same thing.

---

## Finding 26: Distribution Analysis — NODE Token Family Severely Underrepresented

**How found**: Counted actual token distributions in all_labeled_pool.jsonl (1,522 jobs) against V5 plan minimum targets. Then searched ~5,000 real LinkedIn jobs across all available scraping sources.

**Impact**: NODE had 38 examples vs 80 minimum. NODE_AI_ML had 0 examples vs 15 minimum. These are the tokens that produce `good_fit` labels (NODE=10, NODE_AI_ML=20 tech points). V5.1's worst errors were in tech (72.5% accuracy), specifically NODE_JS_TS→JS_TS confusion — the model literally didn't have enough NODE examples to learn the distinction. good_fit labels were also only 91 vs 200 minimum, and maybe only 269 vs 300.

**Data sources searched**:
- `data/scraped_data/` — 14 JSONL files (1,675 jobs)
- `job_search_agent_v2/linkedinScraper/` — 43 JSONL files (318 jobs)
- `job_search_agent_v2/scripts/` — balanced datasets, shopping lists (3,480 jobs)
- `job_search_agent_v2/linkedinScraper/data/` — additional scraper data
- Root `linkedin_jobs.jsonl` files
- `job_search_agent_v2/jobs.db` — SQLite database (92 jobs with descriptions)
- `shoopinglist_5_march_26_v3/` synthetics — checked but rejected (low quality: mismatched company names, template descriptions, some didn't mention target tech)

**Result**: After deduplication against pool and within scraped, quality filtering (>100 char descriptions, valid titles), and removing 570 missing-location jobs (kept only NODE ones): 2,543 new real jobs.

**Gap filling**: Pure NODE and NODE_AI_ML are structurally rare in real data (most Node.js jobs also require React/TypeScript → NODE_JS_TS). Generated 46 variants using tech-swap from real backend Python/Java/Go donor jobs:
- 32 NODE variants: swapped Python→Node.js, Django→Express, Flask→Fastify in real backend JDs
- 14 NODE_AI_ML variants: added Node.js backend references to real AI/ML jobs, kept Python for ML context
- Method: programmatic regex replacement per Critical Rule #6 (not GPT rephrasing)
- Quality checks: no frontend leakage, no old tech remnants, capitalisation varied (Node.js/NodeJS/nodejs)
- Synthetic = 1.8% of total (well under 25% cap)

**Decision**: Use combined pool + scraped + synthetic for V6 labeling. All NODE-family gaps now met or exceeded.

**Edge case coverage**: Verified ≥5 examples for all 6 edge case categories from TRAINING_PLAN_V5.md: Dublin/Ireland=65, Belfast/N.Ireland=36, JD-contradicts-title=152, "node" as network node=60, daily rates=20, mixed currencies (GBP+USD/EUR)=72. No gaps to fill — all edge cases are well represented in the combined pool+scraped dataset.

**Lesson**: Pure NODE is genuinely rare in the job market (~1% of real jobs). Most Node.js roles also require frontend tech. This means the model MUST see targeted examples — natural distribution won't teach NODE vs NONE discrimination. Tech-swap from real donors is higher quality than pure synthetic generation because it preserves realistic JD structure (benefits, EO statements, formatting quirks).

---

## Finding 27: Structured Reasoning Format Redesign

**How found**: Analyzed V5 training data and found 1,048 reasoning-token mismatches where GPT-4o-mini used abbreviated tokens in the reasoning string (e.g., "→ UK" instead of "→ UK_OTHER", "→ LONDON" instead of "→ LONDON_OR_REMOTE"). Since reasoning IS part of training data (included in format-for-mlx.ts assistant response), these abbreviations could confuse the 0.5B student model — it might learn abbreviated tokens from the reasoning and output them as actual tokens.

**Analysis**: For a large model like GPT-4o-mini, these inconsistencies are trivial. But for a 0.5B student, every inconsistency in training data is magnified. The reasoning is generated autoregressively BEFORE the token fields, so abbreviated tokens in reasoning create conflicting attention signals.

**Impact**: HIGH — affects all training data format, student prompt, eval parsing.

**Decision**: Complete format redesign from 5-field to 8-field interleaved structured reasoning:
- OLD: `{"reasoning":"loc: evidence → token. role: evidence → token. tech: ... comp: ...","loc":"...","role":"...","tech":"...","comp":"..."}`
- NEW: `{"loc_reason":"evidence -> TOKEN","loc":"TOKEN","role_reason":"evidence -> TOKEN","role":"TOKEN","tech_reason":"evidence -> TOKEN","tech":"TOKEN","comp_reason":"evidence -> TOKEN","comp":"TOKEN"}`

**Key design choices**:
1. **Interleaved** (reason immediately before its token) — tightest attention anchor for tiny model
2. **ASCII "->"** instead of unicode "→" — better tokenizer handling
3. **Each reason ends with "-> FULL_TOKEN_NAME"** — redundancy reinforces correct answer
4. **Semicolons for ignored items**: "node.js; react ignored -> NODE"
5. **Teacher output = Student training data = Student inference output** (same 8-field format, no conversion)
6. **Eval scores only 4 token fields**, ignores _reason fields

**Safeguards built (3-layer defense)**:
- Layer 1: V6 teacher prompt enforces strict format with 3 worked examples
- Layer 2: `audit-training-data.ts` new "reasoning_token_mismatch" check catches abbreviated tokens
- Layer 3: `format-for-mlx.ts` `normalizeReasoning()` auto-fixes any remaining abbreviations

**Files changed**: `prompts/teacher_v6.txt` (updated output format), `prompts/student_v6.txt` (new file), `src/cli/audit-training-data.ts` (new check), `src/cli/format-for-mlx.ts` (normalization)

**Category**: Design decision

**Status**: Implemented, pending downstream file updates

**Lesson**: For tiny models, reasoning quality matters as much as label quality. Free-form reasoning teaches stylistic variation that wastes model capacity. Structured reasoning with predictable patterns lets the model focus capacity on classification accuracy.

---

## Finding 28: V7 Teacher Prompt Architecture — Richer Token Vocabulary for Better Student Training

**How found**: Iterative prompt design session analyzing the V6 teacher prompt against Perplexity research on optimal teacher prompt design for reasoning models (GPT-4o-mini). Identified several structural weaknesses in V6's field design.

**Problems identified in V6**:

1. **LONDON_OR_REMOTE is a blurry token**: Merges two distinct signals (physical London office vs fully remote anywhere). The student model can't learn nuanced location patterns when fundamentally different situations share one label. A London hybrid job and a fully remote job have different patterns in JD text but produce the same token.

2. **ROLE field overloaded**: Combined two independent questions — "is this an engineering role?" (scope) and "what seniority level?" (seniority). When scope=OUT_OF_SCOPE, seniority was forced to NO_SENIORITY regardless of the title (e.g., "Senior Marketing Manager" → NO_SENIORITY). This created conflicting training signals where "Senior" sometimes matters and sometimes doesn't, with no structural explanation.

3. **Keyword matching fights reasoning**: V6 used exhaustive keyword lists ("senior, staff, principal, lead, head, chief, director, VP, CTO, founding, snr, sr, III, distinguished, manager") which GPT-4o-mini pattern-matched instead of reasoning about. This caused failures on novel phrasings.

4. **MISSING location token is vague**: Doesn't distinguish between "location not stated" and "location is unintelligible garbage". UNKNOWN is more semantically honest.

**V7 design (6 fields, 12 JSON keys)**:

| V6 Field | V7 Field(s) | Change |
|----------|-------------|--------|
| loc (4 tokens) | location (5 tokens) | LONDON_OR_REMOTE → IN_LONDON + FULLY_REMOTE. MISSING → UNKNOWN |
| — | work_arrangement (4 tokens) | NEW: REMOTE, HYBRID, IN_OFFICE, UNKNOWN |
| role (3 tokens) | scope (2 tokens) + seniority (3 tokens) | Split into binary gate + level. SENIOR_PLUS → LEVEL_3, MID_LEVEL → LEVEL_2, NO_SENIORITY → LEVEL_1 |
| tech (8 tokens) | tech (8 tokens) | Unchanged |
| comp (6 tokens) | comp (6 tokens) | Unchanged |

**Scoring translation** (backward-compatible — same label formula):
```python
V7_SCORES = {
    'location': {'IN_LONDON': 25, 'FULLY_REMOTE': 25, 'UK_OTHER': 10, 'OUTSIDE_UK': -50, 'UNKNOWN': 0},
    'work_arrangement': {'REMOTE': 0, 'HYBRID': 0, 'IN_OFFICE': 0, 'UNKNOWN': 0},  # informational only
    'scope': {'IN_SCOPE': None, 'OUT_OF_SCOPE': 0},  # gate — if OUT_OF_SCOPE, seniority=0
    'seniority': {'LEVEL_3': 25, 'LEVEL_2': 15, 'LEVEL_1': 0},
    'tech': {same as V6},
    'comp': {same as V6},
}
# scope=OUT_OF_SCOPE → seniority score = 0 (regardless of seniority token)
# score = max(0, min(100, location + seniority_score + tech + comp))
```

**Benefits**:
1. **Richer training signal**: IN_LONDON vs FULLY_REMOTE teaches student to distinguish office-based vs remote patterns in JD text
2. **Cleaner scope gate**: OUT_OF_SCOPE jobs always score 0 for role component — no confusing "Senior → NO_SENIORITY" training examples
3. **Future flexibility**: work_arrangement scores 0 today but could be scored later without re-labeling
4. **Semantic rules**: "determine the seniority level the title conveys" with examples as guidance (e.g., ...) lets GPT-4o-mini reason about novel phrasings instead of keyword matching
5. **107 lines** (V6 was 186 lines) — more concise despite more fields

**Implementation impact**:
- `eval_student.py` needs V7→score translation function (only code change needed)
- Student prompt updated (just field names + output format template)
- `label-jobs.ts` passes `--prompt prompts/teacher_v7.txt` (no code change)
- `format-for-mlx.ts` needs field name mapping (12 fields instead of 8)
- `audit-training-data.ts` needs V7 token vocabulary
- Contrastive batch design needs minor updates (new token names)

**Decision**: Adopt V7 architecture. All downstream scripts updated as part of Step 5 prep.

**Category**: Architecture redesign

**Status**: ✅ Implemented — V7 teacher prompt saved (`prompts/teacher_v7.txt`). All 10 downstream scripts updated (Step 5 complete). Gap-filling data created (Step 5b complete).

**Lesson**: For teacher prompts targeting reasoning models, semantic descriptions with example guidance ("e.g., senior, staff, principal") produce more consistent results than exhaustive keyword lists. The model reasons about the concept rather than pattern-matching against a closed set. Splitting overloaded fields into independent concerns (scope vs seniority) eliminates contradictory training signals.

---

## Finding 29: V6→V7 Distribution Gap Analysis — FULLY_REMOTE, NODE, NODE_AI_ML (V6-7 Transition)

**Date**: 2026-03-09

**How found**: After completing V7 script updates (Step 5), analyzed how V7's new token vocabulary affects the V5 plan's distribution minimums. V7 split LONDON_OR_REMOTE into IN_LONDON + FULLY_REMOTE, and split ROLE into scope + seniority. The V5 plan had minimums for V6 tokens but not for V7's new fields. Estimated V7 token distributions using text-signal heuristics on the existing pool (1,522 labeled) + scraped data (2,589 unlabeled).

**Problems identified**:

1. **FULLY_REMOTE critical gap**: V7 splits V6's LONDON_OR_REMOTE (est. ~360 jobs) into IN_LONDON (~328, 91%) and FULLY_REMOTE (~30, 8%). The 60-job minimum for FULLY_REMOTE was not met. Root cause: LinkedIn job scrapes store remote status in a separate `workplaceType` field that was consistently empty in our scraped data. The `job_location` text field almost never says "Remote" — it typically shows the company HQ city. This means **zero real fully-remote jobs existed in any data source**.

2. **NODE persistent gap**: Despite 32 synthetic NODE variants created in Step 4 (V6), text-signal estimates showed only ~37 NODE-only jobs in the pool — well below the 80-job minimum. The problem: real backend jobs overwhelmingly use Python, Java, or Go. Node.js backend-only roles (without TypeScript/JavaScript mentioned) are genuinely rare on LinkedIn.

3. **NODE_AI_ML gap**: Estimated ~15 jobs (14 existing V6 synthetic + ~1 real) vs 20-job minimum. NODE combined with AI/ML is a niche combination in real job postings.

4. **No V5 minimums for new fields**: V5 plan had no distribution targets for work_arrangement or scope (they didn't exist). Needed V7-specific targets.

**External data scan results**:
Scanned all available external data sources before creating synthetic variants:

| Source | Jobs | New (not in pool) | Notes |
|--------|------|-------------------|-------|
| `job_searcher/custom_training_data_2batch` | 150 | 0 | All duplicates of pool data |
| `job_searcher/balanced_dataset_raw` | 1,330 | 0 | All duplicates of pool data |
| `job_search_agent_v2/real_linkedin_500_raw` | 980 | 2 | 2 genuinely new jobs |
| `job_search_agent_v2/linkedinScraper` | 317 | 1 | 1 genuinely new job |
| **Total** | ~2,777 | **3** | Almost everything was already imported during Step 4 |

**Conclusion**: External sources are exhausted. Gap-filling must use programmatic variants of real JDs.

**Decisions**:

1. **FULLY_REMOTE**: Created 60 location-swap variants from 60 unique real UK engineering JDs. Changed only the `job_location` field to one of 12 remote formats ("Remote", "Remote (UK)", "Fully Remote", "Remote - United Kingdom", "UK Remote", "Remote, United Kingdom", "Fully Remote (UK based)", "Remote within United Kingdom", "United Kingdom (Remote)", "Work from Home - UK", "UK - Remote", "Remote UK"). JD text unchanged — teaches the student that "Remote" appears in the location field, not the JD body. 5 donors per format for diversity. File: `data/v7/remote_variants.jsonl`.

2. **NODE**: Created 45 additional NODE tech-swap variants from different Python/Java/Go backend donors (not overlapping with V6's 32 existing variants). Regex replacement: primary tech → Node.js, framework → Express.js. Verified all have "Node.js" mentioned, none have "JavaScript" or "TypeScript". File: `data/v7/synthetic/node_variants_v7.jsonl`.

3. **NODE_AI_ML**: Created 10 variants by taking NODE tech-swap variants and appending AI/ML requirement snippets. Each has unique AI/ML text (machine learning pipelines, deep learning frameworks, NLP/LLM, computer vision, MLOps, etc.). File: `data/v7/synthetic/node_ai_ml_variants_v7.jsonl`.

4. **V7 distribution minimums**: Added comprehensive V7 minimum table to V6_STUDENT_TRAINING_PLAN.md (Step 5b). New field targets: work_arrangement (REMOTE≥60, HYBRID≥80, IN_OFFICE≥50, UNKNOWN≥100), scope (IN_SCOPE≥400, OUT_OF_SCOPE≥80). Existing field targets adjusted for V7 token names.

**Impact on data pipeline**:
| Metric | Before | After |
|--------|--------|-------|
| Grand total for labeling | 4,111 | 4,226 |
| Synthetic variants | 46 (1.1%) | 161 (3.8%) |
| Estimated FULLY_REMOTE | ~30 | ~90 |
| Estimated NODE | ~37 | ~82 |
| Estimated NODE_AI_ML | ~15 | ~25 |
| Synthetic cap (25%) | ✅ | ✅ |

**Alternatives considered**:
- **GPT-generate remote JDs**: Rejected — programmatic location-swap from real JDs produces more realistic training data and avoids GPT hallucination artifacts.
- **Scrape remote-specific job boards (e.g., RemoteOK, WeWorkRemotely)**: Rejected — would require new scraping infrastructure and the JD format/style would differ significantly from LinkedIn. Better to teach the model via location field variants of familiar JD structures.
- **Lower FULLY_REMOTE minimum**: Rejected — if FULLY_REMOTE is too rare in training, the model will never learn to distinguish it from IN_LONDON. The 60-job minimum is the floor for pattern learning in a 0.5B model.

**Category**: Data distribution / V6-7 transition

**Status**: ✅ Fixed — all gap-filling data created and ready for V7 labeling. Distribution minimums updated in training plan.

**Lesson**: When redesigning token vocabulary (V6→V7), always check whether field splits create new distribution gaps. LONDON_OR_REMOTE had 360 jobs but FULLY_REMOTE had ~30 — a 92/8 split that was invisible until the field was decomposed. LinkedIn's location data doesn't capture remote status, so this gap cannot be fixed by scraping more data — only by programmatic variants. External data sources should be checked early but don't assume they contain what you need; in this case, 2,777 external jobs yielded only 3 new entries.
