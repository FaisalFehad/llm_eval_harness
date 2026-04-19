#!/usr/bin/env python3
"""
Build V7 Training Dataset
=========================
Curates a high-quality 1,200-job dataset matching exact V7 distribution targets.

Pipeline:
1. Load all data sources (V5 pool, V6 scraped, V7 variants, external DBs)
2. Normalize to common format
3. Deduplicate (exact + fuzzy similarity)
4. Quality filter (min JD length, no garbage)
5. Estimate V7 tokens from available labels + text heuristics
6. Greedy selection to match target distribution
7. Report gaps needing generation
8. Output curated dataset

Usage:
    python3 scripts/build_v7_dataset.py --analyze        # Gap analysis only
    python3 scripts/build_v7_dataset.py --select         # Select + output dataset
    python3 scripts/build_v7_dataset.py --select --output data/v7/curated_dataset.jsonl
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── V7 Distribution Targets ──────────────────────────────────────────────────

DISTRIBUTION_TARGETS = {}

def load_distribution_targets(csv_path: str) -> dict:
    """Load exact distribution targets from CSV."""
    targets = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row['Field'].lower()
            token = row['Token']
            targets[field][token] = {
                'total': int(row['Total']),
                'train': int(row['Train']),
                'val': int(row['Val']),
                'test': int(row['Test']),
            }
    return dict(targets)


# ── V7 Token Estimation ──────────────────────────────────────────────────────

def estimate_v7_location(job: dict) -> str:
    """Estimate V7 location token from raw location field."""
    raw_loc = (job.get('location') or '').lower().strip()

    if not raw_loc or raw_loc in ('n/a', 'na', 'not specified', 'none', ''):
        return 'UNKNOWN'

    # Check for remote first
    remote_keywords = ['remote', 'anywhere', 'work from home', 'wfh', 'distributed']
    is_remote = any(kw in raw_loc for kw in remote_keywords)
    has_london = 'london' in raw_loc

    if has_london:
        return 'IN_LONDON'
    elif is_remote:
        return 'FULLY_REMOTE'

    # UK locations
    uk_cities = [
        'manchester', 'birmingham', 'leeds', 'bristol', 'edinburgh',
        'glasgow', 'cardiff', 'cambridge', 'oxford', 'brighton',
        'nottingham', 'liverpool', 'sheffield', 'newcastle', 'southampton',
        'reading', 'belfast', 'bath', 'exeter', 'york', 'coventry',
        'leicester', 'aberdeen', 'dundee', 'swansea', 'plymouth',
        'warwick', 'norwich', 'portsmouth', 'swindon', 'guildford',
    ]
    uk_regions = [
        'united kingdom', 'england', 'wales', 'scotland', 'northern ireland',
        ', uk', 'uk,', 'uk '
    ]

    if any(city in raw_loc for city in uk_cities):
        return 'UK_OTHER'
    if any(region in raw_loc for region in uk_regions):
        # Could be UK_OTHER or IN_LONDON depending on context
        return 'UK_OTHER'

    # Outside UK indicators
    outside_uk = [
        'united states', 'usa', 'u.s.', 'new york', 'california', 'san francisco',
        'seattle', 'austin', 'chicago', 'boston', 'berlin', 'paris', 'amsterdam',
        'dublin', 'india', 'germany', 'france', 'canada', 'australia',
        'singapore', 'israel', 'spain', 'italy', 'netherlands', 'sweden',
        'denmark', 'norway', 'switzerland', 'japan', 'china', 'brazil',
        'poland', 'warsaw', 'czech', 'romania', 'portugal', 'lisbon',
        ', ca', ', ny', ', tx', ', wa', ', ma', ', il', ', co',
    ]
    if any(loc in raw_loc for loc in outside_uk):
        return 'OUTSIDE_UK'

    # If we can't determine, mark as UNSURE (will need manual review or labeling)
    return 'UNSURE'


def estimate_v7_work_arrangement(job: dict) -> str:
    """Estimate work arrangement from raw location and JD text."""
    raw_loc = (job.get('location') or '').lower()
    jd = (job.get('jd_text') or job.get('description') or '').lower()[:2000]

    if 'hybrid' in raw_loc or '(hybrid)' in raw_loc:
        return 'HYBRID'
    if 'remote' in raw_loc and ('hybrid' not in raw_loc):
        return 'REMOTE'
    if 'on-site' in raw_loc or 'on site' in raw_loc or 'in-office' in raw_loc:
        return 'IN_OFFICE'

    # Check JD text
    if 'hybrid' in jd and ('remote' in jd or 'office' in jd):
        return 'HYBRID'
    if re.search(r'fully\s+remote|100%\s+remote|work\s+from\s+home|work\s+remotely', jd):
        return 'REMOTE'
    if re.search(r'on[- ]site|in[- ]office|office[- ]based|in[- ]person', jd):
        return 'IN_OFFICE'

    return 'UNKNOWN'


def estimate_v7_scope(job: dict) -> str:
    """Estimate V7 scope (IN_SCOPE vs OUT_OF_SCOPE) from title and JD."""
    title = (job.get('title') or '').lower()
    jd = (job.get('jd_text') or job.get('description') or '').lower()[:1000]

    # Out of scope: non-engineering roles
    out_of_scope_patterns = [
        r'\b(product\s+manager|project\s+manager|scrum\s+master|business\s+analyst)\b',
        r'\b(data\s+analyst|data\s+scientist|data\s+engineer)\b',
        r'\b(devops|sre|site\s+reliability|infrastructure)\b',
        r'\b(qa\s+engineer|test\s+engineer|quality\s+assurance)\b',
        r'\b(designer|ux|ui\s+designer|graphic)\b',
        r'\b(recruiter|hr\b|human\s+resources|talent)\b',
        r'\b(sales|marketing|account\s+manager|customer\s+success)\b',
        r'\b(support\s+engineer|it\s+support|helpdesk|help\s+desk)\b',
        r'\b(security\s+engineer|cybersecurity|penetration)\b',
        r'\b(network\s+engineer|systems\s+admin|database\s+admin)\b',
    ]

    for pattern in out_of_scope_patterns:
        if re.search(pattern, title):
            return 'OUT_OF_SCOPE'

    # In scope: software engineering roles
    in_scope_patterns = [
        r'\b(software\s+engineer|software\s+developer|full\s*stack)\b',
        r'\b(frontend|front[- ]end|backend|back[- ]end)\b',
        r'\b(web\s+developer|application\s+developer|app\s+developer)\b',
        r'\b(engineer)\b',  # Generic "engineer" is often software
    ]

    for pattern in in_scope_patterns:
        if re.search(pattern, title):
            return 'IN_SCOPE'

    # Default: check if V5 role exists
    v5_role = job.get('role')
    if v5_role:
        if v5_role in ('SENIOR_PLUS', 'MID_LEVEL'):
            return 'IN_SCOPE'
        elif v5_role == 'NO_SENIORITY':
            # Could be either — need more context
            return 'UNSURE'

    return 'UNSURE'


def estimate_v7_seniority(job: dict) -> str:
    """Estimate V7 seniority from title."""
    title = (job.get('title') or '').lower()

    # LEVEL_3: Senior, Lead, Principal, Staff, Architect, Head, VP, Director
    if re.search(r'\b(senior|sr\.?|snr|lead|principal|staff|architect|head|vp|director|chief|founding|iii|iv)\b', title):
        return 'LEVEL_3'

    # LEVEL_2: Mid-level, no explicit seniority but engineering title
    if re.search(r'\b(mid[- ]?level|intermediate|ii\b|software\s+engineer\b|developer\b|full\s*stack)', title):
        return 'LEVEL_2'

    # LEVEL_1: Junior, entry, intern, graduate, associate
    if re.search(r'\b(junior|jr\.?|entry|intern|graduate|trainee|associate|apprentice|i\b)', title):
        return 'LEVEL_1'

    # V5 fallback
    v5_role = job.get('role')
    if v5_role == 'SENIOR_PLUS':
        return 'LEVEL_3'
    elif v5_role == 'MID_LEVEL':
        return 'LEVEL_2'
    elif v5_role == 'NO_SENIORITY':
        return 'LEVEL_1'

    return 'LEVEL_2'  # Default assumption


def estimate_v7_tech(job: dict) -> str:
    """Estimate V7 tech token from JD text or V5 label."""
    # Use V5 label if available (tech tokens are same in V5 and V7)
    v5_tech = job.get('tech')
    if v5_tech and v5_tech in ('NONE', 'JS_TS', 'NODE', 'NODE_JS_TS', 'AI_ML',
                                'JS_TS_AI_ML', 'NODE_AI_ML', 'NODE_JS_TS_AI_ML'):
        return v5_tech

    # Estimate from JD text
    jd = (job.get('jd_text') or job.get('description') or '').lower()
    title = (job.get('title') or '').lower()
    full_text = title + ' ' + jd

    has_node = bool(re.search(r'\bnode\.?js\b|\bnodejs\b|\bnode\s+js\b', full_text))
    has_js_ts = bool(re.search(r'\bjavascript\b|\btypescript\b|\breact\b|\bangular\b|\bvue\b|\bnext\.?js\b', full_text))
    has_ai_ml = bool(re.search(r'\bmachine\s+learning\b|\bdeep\s+learning\b|\bartificial\s+intelligence\b|\bai\b|\bml\b|\bnlp\b|\bcomputer\s+vision\b|\btensorflow\b|\bpytorch\b|\bllm\b|\bgpt\b|\btransformer\b', full_text))

    if has_node and has_js_ts and has_ai_ml:
        return 'NODE_JS_TS_AI_ML'
    elif has_node and has_ai_ml:
        return 'NODE_AI_ML'
    elif has_node and has_js_ts:
        return 'NODE_JS_TS'
    elif has_js_ts and has_ai_ml:
        return 'JS_TS_AI_ML'
    elif has_node:
        return 'NODE'
    elif has_js_ts:
        return 'JS_TS'
    elif has_ai_ml:
        return 'AI_ML'
    else:
        return 'NONE'


def estimate_v7_comp(job: dict) -> str:
    """Estimate V7 comp token from JD text or V5 label."""
    # Use V5 label if available (comp tokens are same)
    v5_comp = job.get('comp')
    if v5_comp and v5_comp in ('NO_GBP', 'UP_TO_ONLY', 'BELOW_45K',
                                'RANGE_55_74K', 'RANGE_75_99K', 'ABOVE_100K'):
        return v5_comp

    # Check salary field first (external DBs)
    salary = job.get('salary') or ''
    jd = (job.get('jd_text') or job.get('description') or '')
    text = salary + ' ' + jd

    # Look for GBP amounts
    gbp_matches = re.findall(r'£\s?([\d,]+)', text)
    if not gbp_matches:
        return 'NO_GBP'

    amounts = []
    for m in gbp_matches:
        try:
            val = int(m.replace(',', ''))
            if val > 1000:  # Skip hourly/daily rates
                amounts.append(val)
        except ValueError:
            pass

    if not amounts:
        # Has £ but only small amounts (hourly/daily)
        return 'UP_TO_ONLY'

    # Check for "up to" pattern
    if re.search(r'up\s+to\s+£', text.lower()):
        return 'UP_TO_ONLY'

    # Use midpoint for range estimation
    if len(amounts) >= 2:
        midpoint = (min(amounts) + max(amounts)) / 2
    else:
        midpoint = amounts[0]

    if midpoint < 45000:
        return 'BELOW_45K'
    elif midpoint < 55000:
        return 'RANGE_55_74K'  # Conservative — close to boundary
    elif midpoint < 75000:
        return 'RANGE_55_74K'
    elif midpoint < 100000:
        return 'RANGE_75_99K'
    else:
        return 'ABOVE_100K'


def estimate_all_v7_tokens(job: dict) -> dict:
    """Estimate all 6 V7 token fields for a job."""
    return {
        'location': estimate_v7_location(job),
        'work_arrangement': estimate_v7_work_arrangement(job),
        'scope': estimate_v7_scope(job),
        'seniority': estimate_v7_seniority(job),
        'tech': estimate_v7_tech(job),
        'comp': estimate_v7_comp(job),
    }


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str, source_tag: str = None) -> list:
    """Load JSONL file and tag with source."""
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if source_tag:
                j['_source'] = source_tag
            # Normalize description field
            if 'description' in j and 'jd_text' not in j:
                j['jd_text'] = j['description']
            jobs.append(j)
    return jobs


def load_sqlite(db_path: str, source_tag: str) -> list:
    """Load jobs from SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE description IS NOT NULL AND length(description) > 200")
    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for row in rows:
        d = dict(row)
        job = {
            'job_id': f"{source_tag}_{d.get('id', '')}",
            'title': d.get('title', ''),
            'company': d.get('company', ''),
            'location': d.get('location', ''),
            'jd_text': d.get('description', ''),
            'salary': d.get('salary', ''),
            '_source': source_tag,
        }
        jobs.append(job)
    return jobs


def load_all_data(base_dir: str) -> list:
    """Load all available data sources."""
    all_jobs = []

    # 1. V5 labeled pool (has V5 tokens)
    v5_path = os.path.join(base_dir, 'data/v5/all_labeled_pool.jsonl')
    if os.path.exists(v5_path):
        v5 = load_jsonl(v5_path, 'v5_pool')
        print(f"  V5 labeled pool: {len(v5)} jobs")
        all_jobs.extend(v5)

    # 2. V6 scraped (unlabeled)
    v6_path = os.path.join(base_dir, 'data/v6/scraped_clean_for_labeling.jsonl')
    if os.path.exists(v6_path):
        v6 = load_jsonl(v6_path, 'v6_scraped')
        print(f"  V6 scraped: {len(v6)} jobs")
        all_jobs.extend(v6)

    # 3. V7 gap fill files
    v7_files = [
        ('data/v7/remote_variants.jsonl', 'v7_remote'),
        ('data/v7/synthetic/node_variants_v7.jsonl', 'v7_node_synth'),
        ('data/v7/synthetic/node_ai_ml_variants_v7.jsonl', 'v7_node_ai_ml_synth'),
        ('data/v7/gap_fill_real.jsonl', 'v7_gap_real'),
    ]
    for rel_path, tag in v7_files:
        path = os.path.join(base_dir, rel_path)
        if os.path.exists(path):
            jobs = load_jsonl(path, tag)
            print(f"  {tag}: {len(jobs)} jobs")
            all_jobs.extend(jobs)

    # 4. External: job_searcher
    js_db = '/Users/faisal/Code/automation/job_searcher/data/jobs.sqlite'
    if os.path.exists(js_db):
        js = load_sqlite(js_db, 'ext_job_searcher')
        print(f"  External job_searcher: {len(js)} jobs")
        all_jobs.extend(js)

    # 5. External: job_search_agent_v2
    jsa_db = '/Users/faisal/Code/automation/job_search_agent_v2/jobs.db'
    if os.path.exists(jsa_db):
        jsa = load_sqlite(jsa_db, 'ext_agent_v2')
        print(f"  External agent_v2: {len(jsa)} jobs")
        all_jobs.extend(jsa)

    # 6. External: custom training data
    custom_path = '/Users/faisal/Code/automation/job_search_agent_v2/custom_training_data_2batch.jsonl'
    if os.path.exists(custom_path):
        custom = load_jsonl(custom_path, 'ext_custom')
        print(f"  External custom data: {len(custom)} jobs")
        all_jobs.extend(custom)

    return all_jobs


# ── Deduplication ─────────────────────────────────────────────────────────────

def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for deduplication comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def content_hash(job: dict) -> str:
    """Create a content hash from title + location + first 500 chars of JD.

    Includes location so that augmented variants (location-swaps, tech-swaps)
    with different locations but same JD are treated as different jobs.
    """
    title = normalize_text_for_dedup(job.get('title') or '')
    location = normalize_text_for_dedup(job.get('location') or '')
    jd = normalize_text_for_dedup((job.get('jd_text') or '')[:500])
    content = f"{title}||{location}||{jd}"
    return hashlib.md5(content.encode()).hexdigest()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts using word sets."""
    words1 = set(normalize_text_for_dedup(text1).split())
    words2 = set(normalize_text_for_dedup(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def deduplicate(jobs: list, similarity_threshold: float = 0.85) -> tuple:
    """Remove exact and near-duplicate jobs.

    Returns (unique_jobs, removed_count).
    """
    # Phase 1: Exact dedup by job_id
    seen_ids = set()
    phase1 = []
    id_dupes = 0
    for j in jobs:
        jid = j.get('job_id', '')
        if jid and jid in seen_ids:
            id_dupes += 1
            continue
        if jid:
            seen_ids.add(jid)
        phase1.append(j)

    # Phase 2: Content hash dedup
    seen_hashes = set()
    phase2 = []
    hash_dupes = 0
    for j in phase1:
        ch = content_hash(j)
        if ch in seen_hashes:
            hash_dupes += 1
            continue
        seen_hashes.add(ch)
        j['_content_hash'] = ch
        phase2.append(j)

    # Phase 3: Fuzzy dedup (compare JD text similarity for remaining)
    # Use title + location + first 300 chars of JD for comparison
    # This preserves augmented variants where only location/tech changed
    phase3 = []
    fuzzy_dupes = 0
    comparison_texts = []

    for j in phase2:
        title = (j.get('title') or '')
        location = (j.get('location') or '')
        jd = (j.get('jd_text') or '')[:300]
        # Include title and location in comparison so variants survive
        compare_text = f"{title} | {location} | {jd}"
        is_dup = False
        for existing_text in comparison_texts:
            sim = jaccard_similarity(compare_text, existing_text)
            if sim > similarity_threshold:
                is_dup = True
                fuzzy_dupes += 1
                break
        if not is_dup:
            phase3.append(j)
            comparison_texts.append(compare_text)

    total_removed = id_dupes + hash_dupes + fuzzy_dupes
    print(f"\n  Dedup results:")
    print(f"    ID duplicates: {id_dupes}")
    print(f"    Content hash duplicates: {hash_dupes}")
    print(f"    Fuzzy duplicates (>{similarity_threshold}): {fuzzy_dupes}")
    print(f"    Remaining: {len(phase3)}")

    return phase3, total_removed


# ── Quality Filters ───────────────────────────────────────────────────────────

MIN_JD_LENGTH = 200  # Minimum JD text length
MAX_JD_LENGTH = 50000  # Max (avoid mega-documents)

def quality_filter(jobs: list) -> tuple:
    """Filter out low-quality jobs. Returns (good_jobs, removed_count)."""
    good = []
    removed = Counter()

    for j in jobs:
        jd = j.get('jd_text') or ''
        title = j.get('title') or ''

        # Too short
        if len(jd) < MIN_JD_LENGTH:
            removed['too_short'] += 1
            continue

        # Too long (truncate, don't remove)
        if len(jd) > MAX_JD_LENGTH:
            j['jd_text'] = jd[:MAX_JD_LENGTH]

        # No title
        if not title.strip():
            removed['no_title'] += 1
            continue

        # Garbage detection: JD is mostly non-text (e.g., HTML, JSON)
        alpha_ratio = sum(1 for c in jd[:500] if c.isalpha()) / max(len(jd[:500]), 1)
        if alpha_ratio < 0.4:
            removed['garbage_content'] += 1
            continue

        good.append(j)

    print(f"\n  Quality filter results:")
    for reason, count in removed.most_common():
        print(f"    Removed ({reason}): {count}")
    print(f"    Remaining: {len(good)}")

    return good, sum(removed.values())


# ── Distribution-Aware Selection ──────────────────────────────────────────────

def compute_gap(selected_counts: dict, targets: dict) -> dict:
    """Compute gap between current selection and targets."""
    gaps = {}
    for field, token_targets in targets.items():
        gaps[field] = {}
        for token, target_info in token_targets.items():
            current = selected_counts.get(field, {}).get(token, 0)
            target = target_info['total']
            gaps[field][token] = target - current
    return gaps


def job_gap_score(job_tokens: dict, gaps: dict, selected_counts: dict, targets: dict) -> float:
    """Score how well a job fills current gaps. Higher = fills more gaps.

    Strategy:
    - Hard caps on location, tech, comp (fields we can reliably estimate)
    - Soft preferences on work_arrangement, scope, seniority (unreliable estimates)
    - Proportional urgency: rarer tokens get higher priority
    """
    score = 0.0
    hard_penalty_count = 0

    # Fields we can estimate reliably → enforce hard caps
    hard_cap_fields = {'location', 'tech', 'comp'}
    # Fields with unreliable estimates → soft guidance only
    soft_fields = {'work_arrangement', 'scope', 'seniority'}

    for field, token in job_tokens.items():
        if token in ('UNSURE',):
            continue
        if field not in targets or token not in targets.get(field, {}):
            continue

        target = targets[field][token]['total']
        current = selected_counts.get(field, {}).get(token, 0)
        gap = target - current

        if field in hard_cap_fields:
            if gap > 0:
                urgency = gap / max(target, 1)
                score += gap * (1 + urgency)
            else:
                # Hard penalty for over-represented reliable fields
                overshoot = abs(gap)
                score -= 30 + overshoot * 3
                hard_penalty_count += 1
        else:
            # Soft field: mild preference, no hard blocking
            if gap > 0:
                score += gap * 0.5
            else:
                score -= 5  # Mild penalty

    # Extra penalty if too many hard-cap fields are over-filled
    if hard_penalty_count >= 3:
        score -= 150

    return score


def greedy_select(jobs: list, targets: dict, total_target: int = 1200) -> tuple:
    """Greedy selection of jobs to match target distribution.

    Returns (selected_jobs, remaining_gaps).
    """
    # Compute estimated V7 tokens for all jobs
    for j in jobs:
        if '_v7_est' not in j:
            j['_v7_est'] = estimate_all_v7_tokens(j)

    # Filter out jobs with too many UNSURE fields
    usable = [j for j in jobs if sum(1 for v in j['_v7_est'].values() if v == 'UNSURE') <= 2]
    print(f"\n  Usable jobs (≤2 UNSURE fields): {len(usable)}")

    # Initialize selection
    selected = []
    selected_counts = defaultdict(Counter)
    remaining = list(usable)

    # Greedy: pick jobs that fill the most gaps
    iteration = 0
    while len(selected) < total_target and remaining:
        iteration += 1
        gaps = compute_gap(selected_counts, targets)

        # Score all remaining jobs
        scored = []
        for j in remaining:
            score = job_gap_score(j['_v7_est'], gaps, selected_counts, targets)
            scored.append((score, j))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Pick the best job
        best_score, best_job = scored[0]

        if best_score <= -50:
            # All remaining jobs are heavily over-represented, stop
            print(f"  Stopped at {len(selected)} — remaining jobs don't fill gaps")
            break

        selected.append(best_job)
        remaining.remove(best_job)

        # Update counts
        for field, token in best_job['_v7_est'].items():
            if token != 'UNSURE':
                selected_counts[field][token] += 1

        if iteration % 100 == 0:
            print(f"  Selected {len(selected)}/{total_target}...")

    # Final gaps
    final_gaps = compute_gap(selected_counts, targets)

    return selected, final_gaps, dict(selected_counts)


# ── Gap Report ────────────────────────────────────────────────────────────────

def print_gap_report(gaps: dict, selected_counts: dict, targets: dict):
    """Print detailed gap analysis."""
    print("\n" + "=" * 70)
    print("GAP ANALYSIS REPORT")
    print("=" * 70)

    total_gap = 0
    generation_needed = defaultdict(list)

    for field in ['location', 'work_arrangement', 'scope', 'seniority', 'tech', 'comp']:
        if field not in targets:
            continue
        print(f"\n  {field.upper()}")
        print(f"  {'Token':<25} {'Target':>7} {'Have':>7} {'Gap':>7} {'Status'}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

        for token, target_info in targets[field].items():
            target = target_info['total']
            have = selected_counts.get(field, {}).get(token, 0)
            gap = target - have
            status = '✅' if gap <= 0 else f'❌ need {gap}'
            print(f"  {token:<25} {target:>7} {have:>7} {gap:>7} {status}")
            if gap > 0:
                total_gap += gap
                generation_needed[field].append((token, gap))

    print(f"\n  TOTAL GAPS: {total_gap} jobs needed")

    if generation_needed:
        print("\n" + "=" * 70)
        print("GENERATION PLAN")
        print("=" * 70)
        print("\n  Jobs to generate via OpenAI (combine multiple needs per job):")
        for field, tokens in generation_needed.items():
            for token, gap in tokens:
                print(f"    {field}={token}: need {gap} more")

    return generation_needed


# ── Output ────────────────────────────────────────────────────────────────────

def output_dataset(jobs: list, output_path: str):
    """Output curated dataset as JSONL."""
    # Shuffle
    random.seed(42)
    random.shuffle(jobs)

    # Write with normalized fields
    with open(output_path, 'w') as f:
        for j in jobs:
            output = {
                'job_id': j.get('job_id', ''),
                'title': j.get('title', ''),
                'company': j.get('company', ''),
                'location': j.get('location', ''),
                'jd_text': j.get('jd_text', ''),
                '_source': j.get('_source', 'unknown'),
                '_v7_est': j.get('_v7_est', {}),
            }
            # Preserve augmentation info if present
            if 'augmentation_type' in j:
                output['augmentation_type'] = j['augmentation_type']
            if 'source_job_id' in j:
                output['source_job_id'] = j['source_job_id']
            f.write(json.dumps(output) + '\n')

    print(f"\n  Wrote {len(jobs)} jobs to {output_path}")


def output_gap_plan(generation_needed: dict, output_path: str):
    """Output gap generation plan as JSON."""
    plan = {}
    for field, tokens in generation_needed.items():
        plan[field] = {token: gap for token, gap in tokens}

    with open(output_path, 'w') as f:
        json.dump(plan, f, indent=2)

    print(f"  Wrote generation plan to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build V7 Training Dataset')
    parser.add_argument('--analyze', action='store_true', help='Gap analysis only')
    parser.add_argument('--select', action='store_true', help='Select and output dataset')
    parser.add_argument('--output', type=str, default='data/v7/curated_dataset.jsonl',
                        help='Output path for curated dataset')
    parser.add_argument('--gap-plan', type=str, default='data/v7/generation_gap_plan.json',
                        help='Output path for gap generation plan')
    parser.add_argument('--csv', type=str, default='data/v7/distribution_raw.csv',
                        help='Path to distribution targets CSV')
    parser.add_argument('--base-dir', type=str,
                        default='/Users/faisal/Code/automation/ai_eval_harness',
                        help='Base directory for data files')
    args = parser.parse_args()

    # Resolve paths
    csv_path = os.path.join(args.base_dir, args.csv) if not os.path.isabs(args.csv) else args.csv
    output_path = os.path.join(args.base_dir, args.output) if not os.path.isabs(args.output) else args.output
    gap_plan_path = os.path.join(args.base_dir, args.gap_plan) if not os.path.isabs(args.gap_plan) else args.gap_plan

    print("=" * 70)
    print("V7 DATASET BUILDER")
    print("=" * 70)

    # Load targets
    print("\n[1] Loading distribution targets...")
    targets = load_distribution_targets(csv_path)
    total_target = sum(t['total'] for t in targets['location'].values())
    print(f"  Target: {total_target} total jobs")

    # Load all data
    print("\n[2] Loading all data sources...")
    all_jobs = load_all_data(args.base_dir)
    print(f"  Total loaded: {len(all_jobs)} jobs")

    # Deduplicate
    print("\n[3] Deduplicating...")
    unique_jobs, dedup_removed = deduplicate(all_jobs)

    # Quality filter
    print("\n[4] Quality filtering...")
    quality_jobs, quality_removed = quality_filter(unique_jobs)

    # Estimate V7 tokens
    print("\n[5] Estimating V7 tokens for all jobs...")
    for j in quality_jobs:
        j['_v7_est'] = estimate_all_v7_tokens(j)

    # Distribution analysis
    print("\n[6] Current estimated distribution (all available data):")
    all_counts = defaultdict(Counter)
    for j in quality_jobs:
        for field, token in j['_v7_est'].items():
            if token != 'UNSURE':
                all_counts[field][token] += 1

    for field in ['location', 'work_arrangement', 'scope', 'seniority', 'tech', 'comp']:
        print(f"\n  {field.upper()}")
        for token, count in sorted(all_counts[field].items(), key=lambda x: -x[1]):
            target = targets.get(field, {}).get(token, {}).get('total', '?')
            status = '✅' if isinstance(target, int) and count >= target else '⚠️'
            print(f"    {token:<25} have={count:>5}  target={target:>5}  {status}")

    if args.analyze:
        # Just report gaps, don't select
        print("\n[7] Gap analysis (without selection):")
        for field in ['location', 'work_arrangement', 'scope', 'seniority', 'tech', 'comp']:
            if field not in targets:
                continue
            for token, target_info in targets[field].items():
                have = all_counts[field].get(token, 0)
                gap = target_info['total'] - have
                if gap > 0:
                    print(f"    {field}={token}: need {gap} more (have {have}, target {target_info['total']})")
        return

    if args.select:
        # Greedy selection
        print("\n[7] Greedy selection to match targets...")
        selected, final_gaps, selected_counts = greedy_select(quality_jobs, targets, total_target)

        # Report
        generation_needed = print_gap_report(final_gaps, selected_counts, targets)

        # Output
        print("\n[8] Outputting...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_dataset(selected, output_path)

        if generation_needed:
            output_gap_plan(generation_needed, gap_plan_path)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total available: {len(all_jobs)}")
        print(f"  After dedup: {len(unique_jobs)}")
        print(f"  After quality: {len(quality_jobs)}")
        print(f"  Selected: {len(selected)}")
        total_gap = sum(gap for field_gaps in final_gaps.values()
                        for gap in field_gaps.values() if gap > 0)
        print(f"  Remaining gap: {total_gap} jobs to generate")

        # Source breakdown
        source_counts = Counter(j.get('_source', 'unknown') for j in selected)
        print(f"\n  Source breakdown of selected jobs:")
        for src, count in source_counts.most_common():
            print(f"    {src}: {count}")


if __name__ == '__main__':
    main()
