#!/usr/bin/env python3
"""Re-label only the 61 jobs with genuine bugs, using the updated prompt."""
import json, os, sys
from pathlib import Path

from openai import OpenAI

# Load bug indices
with open("versions/v16/data/rule_audit_latest.json") as f:
    report = json.load(f)
bug_indices = set(i["index"] for i in report["genuine_issues"])

# Load full dataset
with open("versions/v16/data/v16_teacher_labels.jsonl") as f:
    full_jobs = [json.loads(l) for l in f]

# Find bug jobs and non-bug jobs
bug_jobs = [j for j in full_jobs if j.get("index") in bug_indices]
unchanged_jobs = [j for j in full_jobs if j.get("index") not in bug_indices]

print(f"Re-labeling {len(bug_jobs)} buggy jobs; keeping {len(unchanged_jobs)} unchanged.")

# Read updated prompt
with open("versions/v16/prompts/teacher.txt") as f:
    prompt_text = f.read()

# Setup API client
api_key = "fbe900938b6dedf7eb6994a03054ebbe1654b7d6788c35fdccde7568c8c70e76f300f3faebec1ff7e7066b45a07b7920d406a445682c9019ff71f9cae4d07c17ee473c670c2442d44b04d1b3f4f74f650453d0e0b7743567a525cd3c34f7df5553c64f96fb940b12ab15a8b5d094b5a53dbf7daf9a81fef711b55f7ff2b07c35"
client = OpenAI(base_url="http://localhost:11434/v1", api_key=api_key)  # Use fast Ollama
MODEL = "gemma4:31b-cloud"

def relabel_job(job):
    """Send one job to teacher API with updated prompt."""
    system_msg = "Respond with JSON only. No markdown, no code blocks."
    user_prompt = prompt_text.replace("{{job_title}}", job.get("title", ""))
    user_prompt = user_prompt.replace("{{job_location}}", job.get("job_location", ""))
    user_prompt = user_prompt.replace("{{jd_text}}", job.get("jd_text", ""))

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response")
            # Strip markdown fences
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            import time; time.sleep(2 ** attempt)
    return None

# Write output
output_path = "versions/v16/data/v16_teacher_labels_corrected.jsonl"
with open(output_path, "w") as out:
    # First write all unchanged jobs
    for job in unchanged_jobs:
        out.write(json.dumps(job) + "\n")

    # Then re-label bug jobs
    for i, job in enumerate(bug_jobs):
        print(f"[{i+1}/{len(bug_jobs)}] Re-labeling: {job.get('title','')[:40]}")
        new_labels = relabel_job(job)
        if new_labels:
            # Update v16 fields
            job["v16_loc_raw"] = new_labels.get("loc_raw")
            job["v16_loc"] = new_labels.get("loc")
            job["v16_arr_raw"] = new_labels.get("arr_raw")
            job["v16_arr"] = new_labels.get("arr")
            job["v16_sen_raw"] = new_labels.get("sen_raw")
            job["v16_sen"] = new_labels.get("sen")
            job["v16_tech_raw"] = new_labels.get("tech_raw")
            job["v16_tech"] = new_labels.get("tech")
            job["v16_comp_raw"] = new_labels.get("comp_raw")
            job["v16_comp"] = new_labels.get("comp")
            print(f"  -> New labels: loc={new_labels.get('loc')}, sen={new_labels.get('sen')}, comp={new_labels.get('comp')}")
        else:
            print(f"  -> FAILED to re-label, keeping original")

        out.write(json.dumps(job) + "\n")

print(f"\nCorrected dataset saved to: {output_path}")
print(f"Total jobs: {len(unchanged_jobs) + len(bug_jobs)} = {len(full_jobs)}")

# Quick diff
print("\n=== Bug fixes applied ===")
for job in bug_jobs[:10]:
    title = job.get('title','')[:40]
    print(f"  {title}")
