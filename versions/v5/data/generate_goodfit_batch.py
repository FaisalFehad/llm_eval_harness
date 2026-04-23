"""
Generate 45 targeted good_fit synthetic jobs to fill underrepresented combinations.
Outputs: data/v5/synthetic_goodfit.jsonl
Run: python3 data/v5/generate_goodfit_batch.py
"""
import json
import os
import time
from openai import OpenAI

client = OpenAI()

# Location format helper — used in prompts to ensure city-level diversity
LONDON_FORMATS = [
    "London, United Kingdom", "London, England", "London (Hybrid)",
    "London, England, United Kingdom", "London (Remote)",
]
REMOTE_FORMATS = [
    "Remote - UK", "UK Remote", "Remote, United Kingdom",
    "United Kingdom (Remote)", "Fully Remote - UK",
]
UK_OTHER_CITIES = [
    "Manchester, England", "Bristol, England", "Edinburgh, Scotland",
    "Birmingham, England", "Leeds, England", "Glasgow, Scotland",
    "Cardiff, Wales", "Nottingham, England", "Sheffield, England",
    "Cambridge, England", "Oxford, England", "Brighton, England",
]

RECIPES = [
    # (recipe_id, count, prompt)
    ("P1", 8, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of these formats (vary them): "London, United Kingdom", "London, England, United Kingdom", "London (Hybrid)", "London, England"
- Title: MUST contain "Senior", "Lead", or "Principal"
- Tech: MUST mention Node.js AND TypeScript as REQUIRED skills. Also mention React or GraphQL (these are ignored in scoring). No AI/ML.
- Salary: MUST include a GBP range with midpoint ≥ £100k (e.g. "£95,000-£120,000", "£100k-£130k", "£110,000 per annum")
- jd_text: 200-400 words. Company description (2-3 sentences), requirements, benefits, equal opportunity. Vary Node.js capitalisation (node.js / NodeJS / nodejs). At least one misleading detail (e.g. mention a non-GBP salary for a US office alongside the GBP salary)."""),

    ("P2", 7, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of these (vary): "Remote - UK", "UK Remote", "United Kingdom (Remote)", "Remote, United Kingdom", "Fully Remote - UK"
- Title: MUST contain "Senior", "Staff", or "Lead"
- Tech: MUST mention Node.js (or NodeJS/nodejs) AND AI/ML/LLM as REQUIRED (not nice-to-have). Do NOT separately mention JavaScript or TypeScript as scored tech. Can mention Python as additional.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£70,000-£90,000", "£75k-£95k", "£80,000-£100,000")
- jd_text: 200-400 words. Include: company description, equal opportunity statement, benefits. The AI/ML requirement must be clear in the requirements section, not just mentioned in passing."""),

    ("P3", 5, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "London, United Kingdom", "London (Remote)"
- Title: MUST contain "Senior", "Lead", or "Principal"
- Tech: MUST mention Node.js AND AI/ML/LLM as REQUIRED skills. Do NOT separately mention JavaScript or TypeScript as scored tech.
- Salary: MUST include a GBP salary with midpoint ≥ £100k (e.g. "£100,000-£130,000", "£110k-£140k", "£120,000 per annum")
- jd_text: 200-400 words. This is an AI/backend engineering role. The LLM/AI requirement should be prominent in the requirements section. Include benefits, company description."""),

    ("P4", 5, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "London, United Kingdom", "Remote - UK", "UK Remote"
- Title: MUST contain "Senior", "Lead", "Staff", or "Principal"
- Tech: MUST mention Node.js AND TypeScript AND AI/ML/LLM as REQUIRED. This is an AI-focused full-stack backend role.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£75,000-£95,000", "£80k-£100k")
- jd_text: 200-400 words. AI/ML requirement must be explicit in requirements, not just company description. Vary Node.js capitalisation."""),

    ("P5", 4, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "London, United Kingdom", "Remote - UK"
- Title: MUST contain "Full Stack", "Full-Stack", or "Mid-Level" — NOT Senior or Lead. A well-paid mid-level role.
- Tech: MUST mention Node.js AND TypeScript/JavaScript as REQUIRED. No AI/ML.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£70,000-£90,000", "£75k-£95k"). A high-paying mid-level role.
- jd_text: 200-400 words. Include company description, benefits, equal opportunity statement."""),

    ("P6", 4, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "London, United Kingdom", "Remote - UK", "London (Hybrid)"
- Title: MUST contain "Senior" or "Lead"
- Tech: MUST mention TypeScript or JavaScript as REQUIRED. Do NOT mention Node.js at all. Can mention React, Vue, or Angular as additional.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£70,000-£90,000", "£75k-£95k")
- jd_text: 200-400 words. A frontend-focused role. Include company description, benefits, equal opportunity statement."""),

    ("P7", 4, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: MUST be a UK city that is NOT London and NOT Remote. Use specific city names (vary them): "Manchester, England", "Bristol, England", "Edinburgh, Scotland", "Birmingham, England", "Leeds, England", "Glasgow, Scotland", "Cambridge, England", "Oxford, England"
- Title: MUST contain "Senior", "Lead", or "Principal"
- Tech: MUST mention Node.js AND TypeScript/JavaScript as REQUIRED. No AI/ML needed.
- Salary: MUST include a GBP salary with midpoint ≥ £100k (e.g. "£95,000-£120,000", "£100k-£130k"). High-paying provincial role.
- jd_text: 200-400 words. Include company description, benefits, equal opportunity statement. Mention the specific city in the description."""),

    ("P8", 3, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "Remote - UK", "London, United Kingdom"
- Title: MUST contain "Senior", "Lead", or "Head of"
- Tech: MUST mention Node.js AND TypeScript AND AI/ML/LLM as REQUIRED. This is a senior AI engineering role.
- Salary: MUST include a GBP salary with midpoint ≥ £100k (e.g. "£110,000-£140,000", "£120k+", "£115,000 per annum")
- jd_text: 200-400 words. AI/ML must be explicitly required in the requirements section. Include benefits, company description."""),

    ("P9", 3, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "London, United Kingdom", "Remote - UK", "London (Hybrid)"
- Title: MUST contain "Senior" or "Lead"
- Tech: MUST mention Node.js AND TypeScript/JavaScript as REQUIRED. No AI/ML.
- Salary: MUST include a GBP range with midpoint exactly £55k-£74k (e.g. "£55,000-£70,000", "£60k-£75k", "£65,000 per annum"). This is a good_fit borderline — the salary is lower than ideal but everything else scores.
- jd_text: 200-400 words. Include company description, benefits, equal opportunity statement."""),

    ("P10", 2, """Generate a realistic job posting as JSON: {"title":"...","company":"...","location":"...","jd_text":"..."}.

Requirements:
- Location: one of (vary): "London, England", "Remote - UK", "London, United Kingdom"
- Title: MUST contain "Senior" or "Lead"
- Tech: MUST mention AI/ML/LLM as REQUIRED (not nice-to-have). Do NOT mention Node.js or TypeScript/JavaScript as scored tech. Can mention Python, FastAPI as additional.
- Salary: MUST include a GBP salary with midpoint ≥ £100k (e.g. "£100,000-£130,000", "£110k-£140k"). AI/ML specialist role.
- jd_text: 200-400 words. The AI/ML requirement must be prominent. Include company description, benefits."""),
]

# bad_fit jobs with diverse OUTSIDE_UK locations the model hasn't seen
# Each city is injected dynamically per call — see generate_job_with_city()
BADFIT_CITIES = [
    ("São Paulo, Brazil", "BRL"),
    ("Tokyo, Japan", "JPY"),
    ("Cape Town, South Africa", "ZAR"),
    ("Dubai, United Arab Emirates", "AED"),
    ("Stockholm, Sweden", "SEK"),
    ("Kraków, Poland", "PLN"),
    ("Seoul, South Korea", "KRW"),
    ("Mexico City, Mexico", "MXN"),
    ("Bangalore, India", "INR"),
    ("Jakarta, Indonesia", "IDR"),
]

BADFIT_PROMPT_TEMPLATE = """Generate a realistic job posting as JSON: {{"title":"...","company":"...","location":"...","jd_text":"..."}}.

Requirements:
- Location: MUST be "{{CITY}}" or just the city name without country (e.g. "{{CITY_SHORT}}"). Vary the format.
- Title: any software engineering title — Senior Engineer, Software Developer, Backend Engineer, etc.
- Tech: any realistic tech stack for this role — Python, Java, Go, React, Node.js, TypeScript. Include what makes sense.
- Salary: use the local currency ({{CURRENCY}}) or no salary. Do NOT use GBP.
- jd_text: 200-400 words. Realistic local job posting for this city. Include company description, benefits, local context."""

def generate_job(recipe_id: str, prompt: str, idx: int) -> dict | None:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate realistic job postings. Output ONLY valid JSON, no markdown fences, no extra text."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            job = json.loads(content)
            job["job_id"] = f"synth_goodfit_{recipe_id}_{idx:03d}"
            job["augmentation_type"] = "good_fit_targeted"
            job["source_file"] = "synthetic_v5"
            return job
        except json.JSONDecodeError as e:
            print(f"    Parse error attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error attempt {attempt+1}: {e}")
                time.sleep(1)
    return None

def main():
    output_path = "data/v5/synthetic_goodfit.jsonl"
    gf_total = sum(count for _, count, _ in RECIPES)
    bf_total = len(BADFIT_CITIES)
    total = gf_total + bf_total
    print(f"Generating {total} synthetic jobs ({gf_total} good_fit + {bf_total} bad_fit OUTSIDE_UK)")
    print(f"Output: {output_path}")
    print("─" * 60)

    all_jobs = []

    # good_fit recipes
    for recipe_id, count, prompt in RECIPES:
        print(f"\nRecipe {recipe_id}: {count} jobs [good_fit]")
        for i in range(count):
            job = generate_job(recipe_id, prompt, i + 1)
            if job:
                all_jobs.append(job)
                print(f"  [{len(all_jobs):2d}] {job.get('title','?')[:55]} | {job.get('location','?')}")
            else:
                print(f"  [FAILED] Recipe {recipe_id} idx {i+1}")

    # bad_fit OUTSIDE_UK — one city per call, injected dynamically
    print(f"\nBadFit OUTSIDE_UK: {bf_total} jobs (one per city)")
    for i, (city, currency) in enumerate(BADFIT_CITIES):
        city_short = city.split(",")[0]
        prompt = (BADFIT_PROMPT_TEMPLATE
                  .replace("{{CITY}}", city)
                  .replace("{{CITY_SHORT}}", city_short)
                  .replace("{{CURRENCY}}", currency))
        job = generate_job(f"BF_{i+1}", prompt, i + 1)
        if job:
            all_jobs.append(job)
            print(f"  [{len(all_jobs):2d}] {job.get('title','?')[:45]} | {job.get('location','?')}")
        else:
            print(f"  [FAILED] BF city={city}")

    with open(output_path, "w") as f:
        for job in all_jobs:
            f.write(json.dumps(job) + "\n")

    print(f"\n{'─'*60}")
    print(f"Generated {len(all_jobs)}/{total} jobs → {output_path}")

if __name__ == "__main__":
    main()
