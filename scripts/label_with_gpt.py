#!/usr/bin/env python3
import os, json, argparse, time
from pathlib import Path
import openai

SYSTEM = """You are a precise job-fit classifier. Output strict JSON with keys loc, arr, sen, tech, comp.
loc: IN_LONDON|REMOTE|UK_OTHER|OUTSIDE_UK|UNK
arr: REMOTE|HYBRID|IN_OFFICE|UNK
sen: LEVEL_3|LEVEL_2|LEVEL_1
tech: JSON array using NODE, REACT, JS_TS, AI_ML, OOS (use ["OOS"] if none)
comp: NO_GBP|UP_TO_ONLY|BELOW_45K|RANGE_45_54K|RANGE_55_74K|RANGE_75_99K|ABOVE_100K
Return only JSON, no text."""

USER_TMPL = """Title: {title}
Location: {location}
Description: {jd}

{{"loc":"...","arr":"...","sen":"...","tech":[],"comp":"..."}}"""

def call(client, prompt):
    last_err=None
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role":"system","content":SYSTEM}, {"role":"user","content":prompt}],
                temperature=0,
                max_tokens=200,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err=e
            time.sleep(1)
    if last_err:
        print("API error", last_err)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip', type=int, default=0)
    args = ap.parse_args()

    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    out = open(args.output,'a')
    with open(args.input) as f:
        for idx,line in enumerate(f,1):
            if args.skip and idx<=args.skip:
                continue
            if not line.strip():
                continue
            if args.limit and idx>args.limit:
                break
            job=json.loads(line)
            prompt = USER_TMPL.format(title=job.get('title',''), location=job.get('job_location',job.get('location','')), jd=job.get('jd_text',''))
            txt = call(client,prompt)
            if not txt:
                continue
            try:
                parsed=json.loads(txt)
            except Exception:
                continue
            job['loc']=parsed.get('loc')
            job['arr']=parsed.get('arr')
            job['sen']=parsed.get('sen')
            job['tech']=parsed.get('tech')
            job['comp']=parsed.get('comp')
            # simple scores via semantic_tokens_v7 compatible logic
            # approximate
            loc_score={'IN_LONDON':25,'UK_OTHER':10,'REMOTE':15,'OUTSIDE_UK':-50,'UNK':0}.get(job['loc'],0)
            role_score={'LEVEL_3':25,'LEVEL_2':15,'LEVEL_1':0}.get(job['sen'],0)
            comp_map={'ABOVE_100K':25,'RANGE_75_99K':15,'RANGE_55_74K':5,'RANGE_45_54K':0,'UP_TO_ONLY':0,'BELOW_45K':-30,'NO_GBP':0}
            comp_score=comp_map.get(job['comp'],0)
            tech_score=0
            t=job['tech'] or []
            if 'NODE'in t: tech_score+=10
            if 'JS_TS'in t or 'REACT'in t: tech_score+=5
            if 'AI_ML'in t: tech_score+=10
            tech_score=min(25,tech_score)
            total=max(0,min(100,loc_score+role_score+comp_score+tech_score))
            if total>=70: label='good_fit'
            elif total>=50: label='maybe'
            else: label='bad_fit'
            job['score']=total
            job['label']=label
            job['source_file']='gpt_labeled'
            out.write(json.dumps(job)+'\n')
    out.close()

if __name__=='__main__':
    main()
