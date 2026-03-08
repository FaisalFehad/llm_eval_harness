"""
Generate 30 synthetic patch jobs:
  - 20 garbage-location jobs (loc=MISSING)
  - 10 AI/ML edge cases (tech should NOT include AI_ML)

Output: data/v5/synthetic_patch_raw.jsonl
Run: OPENAI_API_KEY=... python3 data/v5/generate_patch_synthetic.py
"""
import json
import os
import random
import time
from openai import OpenAI

client = OpenAI()

# ── Garbage location strings (all should label MISSING) ──────────────────────

GARBAGE_LOCATIONS = [
    "",
    "Not specified",
    "See description",
    "Various locations",
    "Flexible",
    "TBD",
    "Multiple offices",
    "We are a global company with offices worldwide",
    "Innovative fintech startup headquartered across multiple regions",
    "Our team works where talent is",
    "N/A",
    "Ask recruiter",
    "To be confirmed",
    "Global",
    "Anywhere",
    "Company HQ",
    "Undisclosed",
    "Open to discussion",
    "Please see job description for location details",
    "Distributed team",
]

# ── UK cities for garbage-location job content ────────────────────────────────

UK_CITIES = [
    "London", "Manchester", "Bristol", "Edinburgh", "Birmingham",
    "Leeds", "Glasgow", "Cardiff", "Nottingham", "Sheffield",
]

TECH_STACKS = ["React", "Python", "Go", "Java", "C#/.NET", "Ruby on Rails"]

# ── Recipes ───────────────────────────────────────────────────────────────────

MISSING_LOC_RECIPES = [
    {
        "id": "ML1",
        "count": 4,
        "title_pool": [
            "Senior Software Engineer", "Backend Engineer", "Full Stack Developer",
            "Platform Engineer", "Software Developer",
        ],
        "stack": "Python and Django",
        "seniority": "senior",
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
The role is {seniority} level.
The company is a UK-based fintech or SaaS startup.
Do NOT mention any city or specific location in the job description itself.
Length: 120-180 words.
Output JSON only:
{{"title": "...", "company": "...", "jd_text": "..."}}""",
    },
    {
        "id": "ML2",
        "count": 4,
        "title_pool": [
            "Java Developer", "Backend Java Engineer", "Spring Boot Developer",
            "Senior Java Engineer", "Java Software Engineer",
        ],
        "stack": "Java and Spring Boot",
        "seniority": "mid-level",
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
The role is {seniority} level.
The company is a UK-based enterprise or consultancy.
Do NOT mention any city or specific location in the job description itself.
Length: 120-180 words.
Output JSON only:
{{"title": "...", "company": "...", "jd_text": "..."}}""",
    },
    {
        "id": "ML3",
        "count": 4,
        "title_pool": [
            "PHP Developer", "PHP Laravel Developer", "Backend Developer",
            "Senior PHP Engineer", "Web Developer",
        ],
        "stack": "PHP and Laravel",
        "seniority": "mid-level",
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
The role is {seniority} level.
The company is a UK-based digital agency or e-commerce business.
Do NOT mention any city or specific location in the job description itself.
Length: 120-180 words.
Output JSON only:
{{"title": "...", "company": "...", "jd_text": "..."}}""",
    },
    {
        "id": "ML4",
        "count": 4,
        "title_pool": [
            "Go Engineer", "Golang Developer", "Backend Go Engineer",
            "Senior Golang Engineer", "Platform Engineer",
        ],
        "stack": "Go (Golang)",
        "seniority": "senior",
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
The role is {seniority} level.
The company is a UK-based infrastructure or cloud company.
Do NOT mention any city or specific location in the job description itself.
Length: 120-180 words.
Output JSON only:
{{"title": "...", "company": "...", "jd_text": "..."}}""",
    },
    {
        "id": "ML5",
        "count": 4,
        "title_pool": [
            ".NET Developer", "C# Engineer", "Senior .NET Developer",
            ".NET Software Engineer", "C# Backend Developer",
        ],
        "stack": "C# and .NET",
        "seniority": "senior",
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
The role is {seniority} level.
The company is a UK-based financial services or healthcare tech firm.
Do NOT mention any city or specific location in the job description itself.
Length: 120-180 words.
Output JSON only:
{{"title": "...", "company": "...", "jd_text": "..."}}""",
    },
]

AIML_EDGE_RECIPES = [
    # 7 jobs: company is "AI-powered" but role does NOT require AI/ML
    {
        "id": "AE1",
        "count": 4,
        "prompt": """Write a realistic LinkedIn job description for a {title} role at an AI-powered company.
The company builds AI products (e.g. "AI-driven analytics platform", "machine learning SaaS").
BUT: this specific role is for {stack} backend engineering. The role does NOT require the candidate to have AI or ML skills — they maintain infrastructure, APIs, or data pipelines.
The company description should clearly mention AI. The job requirements must NOT mention AI/ML/LLM as a candidate requirement.
Phrases like "we use AI" or "AI-powered platform" are fine in company description. But requirements like "experience with ML", "LLM experience required", "AI background needed" are NOT allowed.
Location: pick a real UK city (London, Manchester, Bristol, Edinburgh, etc.).
Length: 150-200 words.
Output JSON only:
{{"title": "...", "company": "...", "location": "...", "jd_text": "..."}}""",
        "title_pool": [
            "Backend Engineer", "Platform Engineer", "Senior Software Engineer",
            "Data Engineer", "API Developer",
        ],
        "stack_pool": ["Node.js", "Python", "Go", "Java", "C#/.NET"],
    },
    # 3 jobs: AI/ML is "nice to have" not required
    {
        "id": "AE2",
        "count": 3,
        "prompt": """Write a realistic LinkedIn job description for a {title} role.
Tech stack: {stack}.
Include a "Nice to have" or "Bonus" section that mentions AI/ML/LLM experience as a bonus — NOT required.
The required skills section must NOT list AI/ML/LLM as mandatory.
Use phrases like "Nice to have: experience with ML models", "Bonus: familiarity with LLMs", "Desirable: AI background".
Location: pick a real UK city.
Length: 150-200 words.
Output JSON only:
{{"title": "...", "company": "...", "location": "...", "jd_text": "..."}}""",
        "title_pool": [
            "Senior Software Engineer", "Full Stack Developer", "Backend Developer",
        ],
        "stack_pool": ["Node.js and TypeScript", "Python and Django", "Go"],
    },
]


def call_api(prompt: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            print(f"  Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)
    return None


def generate_missing_loc_jobs() -> list[dict]:
    jobs = []
    loc_iter = iter(GARBAGE_LOCATIONS)

    for recipe in MISSING_LOC_RECIPES:
        for i in range(recipe["count"]):
            title = random.choice(recipe["title_pool"])
            prompt = recipe["prompt"].format(
                title=title,
                stack=recipe["stack"],
                seniority=recipe["seniority"],
            )
            print(f"  [{recipe['id']}-{i+1}] Generating: {title}...")
            result = call_api(prompt)
            if not result:
                print(f"  FAILED {recipe['id']}-{i+1}, skipping")
                continue

            # Inject garbage location — rotate through list
            garbage_loc = next(loc_iter, random.choice(GARBAGE_LOCATIONS))
            job_id = f"synthetic_patch_ml_{recipe['id'].lower()}_{i+1}"

            jobs.append({
                "job_id": job_id,
                "title": result.get("title", title),
                "company": result.get("company", "Company"),
                "location": garbage_loc,
                "jd_text": result.get("jd_text", ""),
                "source_file": "synthetic_patch_v5",
                "batch": "BATCH_MISSING_LOC",
            })
            time.sleep(0.3)

    return jobs


def generate_aiml_edge_jobs() -> list[dict]:
    jobs = []
    for recipe in AIML_EDGE_RECIPES:
        for i in range(recipe["count"]):
            title = random.choice(recipe["title_pool"])
            stack = random.choice(recipe["stack_pool"])
            prompt = recipe["prompt"].format(title=title, stack=stack)
            print(f"  [{recipe['id']}-{i+1}] Generating: {title} ({stack})...")
            result = call_api(prompt)
            if not result:
                print(f"  FAILED {recipe['id']}-{i+1}, skipping")
                continue

            job_id = f"synthetic_patch_ae_{recipe['id'].lower()}_{i+1}"
            jobs.append({
                "job_id": job_id,
                "title": result.get("title", title),
                "company": result.get("company", "Company"),
                "location": result.get("location", "London, United Kingdom"),
                "jd_text": result.get("jd_text", ""),
                "source_file": "synthetic_patch_v5",
                "batch": "BATCH_AIML_EDGE",
            })
            time.sleep(0.3)

    return jobs


def main():
    output_path = "data/v5/synthetic_patch_raw.jsonl"

    print("=== Generating MISSING LOC batch (20 jobs) ===")
    missing_loc = generate_missing_loc_jobs()
    print(f"Generated: {len(missing_loc)}")

    print("\n=== Generating AI/ML EDGE batch (10 jobs) ===")
    aiml_edge = generate_aiml_edge_jobs()
    print(f"Generated: {len(aiml_edge)}")

    all_jobs = missing_loc + aiml_edge
    print(f"\nTotal: {len(all_jobs)} jobs")

    with open(output_path, "w") as f:
        for job in all_jobs:
            f.write(json.dumps(job) + "\n")

    print(f"Written to: {output_path}")

    # Quick QC
    short = [j for j in all_jobs if len(j["jd_text"].split()) < 50]
    if short:
        print(f"WARNING: {len(short)} jobs have <50 words in jd_text")
        for j in short:
            print(f"  {j['job_id']}: {len(j['jd_text'].split())} words")


if __name__ == "__main__":
    main()
