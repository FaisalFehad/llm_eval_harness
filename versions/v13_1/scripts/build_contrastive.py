"""
Build contrastive training examples for V13.1 corrective retrain.

Targets 4 error clusters identified in V13 sweep analysis:
  A: Lead/Staff in title → model misses L3
  B: Manager titles → model predicts L2 (no "manager" in prompt L3 list)
  C: Internship / Roman numeral I → model misses L1
  D: Bare "Full Stack Engineer" → model over-predicts L3

Output: data/v13_1/contrastive.jsonl
"""

import json
from pathlib import Path

SCORE_LOC  = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SCORE_SEN  = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
SCORE_TECH = {"NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10, "OOS": 0}
SCORE_COMP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30,
    "RANGE_45_54K": 0, "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def label_from_score(score):
    if score >= 70: return "good_fit"
    if score >= 50: return "maybe"
    return "bad_fit"


def make(job_id, title, company, job_location, jd_text,
         loc, arr, sen, tech, comp,
         loc_raw, arr_raw, sen_raw, tech_raw, comp_raw,
         cluster):
    is_oos = tech == ["OOS"]
    loc_score  = SCORE_LOC[loc]
    role_score = 0 if is_oos else SCORE_SEN[sen]
    tech_score = 0 if is_oos else sum(SCORE_TECH[t] for t in tech)
    comp_score = SCORE_COMP[comp]
    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "job_location": job_location,
        "jd_text": jd_text,
        "loc_raw": loc_raw, "loc": loc,
        "arr_raw": arr_raw, "arr": arr,
        "sen_raw": sen_raw, "sen": sen,
        "tech_raw": tech_raw, "tech": tech,
        "comp_raw": comp_raw, "comp": comp,
        "loc_score": loc_score,
        "role_score": role_score,
        "tech_score": tech_score,
        "comp_score": comp_score,
        "score": score,
        "label": label_from_score(score),
        "source_file": f"contrastive_v13_1_{cluster}",
    }


examples = [

    # ── Cluster A: Lead / Staff in title → LEVEL_3 ─────────────────────────

    make("v13_1_A1", "Staff Engineer", "Monzo", "London, England, United Kingdom",
         "We're looking for a Staff Engineer to help shape our backend architecture. "
         "You'll work across teams to define technical strategy, mentor engineers, and "
         "own complex infrastructure challenges. This role requires deep technical expertise "
         "and strong cross-team influence. Requirements: 8+ years of engineering experience, "
         "strong knowledge of distributed systems, experience leading technical roadmaps.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="hybrid working — 3 days in our London office",
         sen_raw="Staff Engineer",
         tech_raw="Node.js, distributed systems, Kafka, PostgreSQL",
         comp_raw="£90,000 – £110,000",
         comp="ABOVE_100K", cluster="A"),

    make("v13_1_A2", "Tech Lead", "Deliveroo", "London, England, United Kingdom",
         "Deliveroo is hiring a Tech Lead to own the delivery of our payments platform. "
         "You will lead a squad of 5 engineers, run sprint planning, and own technical "
         "decisions for a high-traffic service processing millions of transactions daily. "
         "You'll be hands-on: writing code, reviewing PRs, and setting architectural direction.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE", "REACT"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid – 2 days in office per week",
         sen_raw="Tech Lead",
         tech_raw="Node.js, React, TypeScript, AWS, PostgreSQL",
         comp_raw="£85,000 – £100,000",
         comp="RANGE_75_99K", cluster="A"),

    make("v13_1_A3", "Lead Developer", "Equal Experts", "London, England, United Kingdom",
         "We are seeking a Lead Developer to join our growing consultancy practice. "
         "You will act as technical lead on client engagements, guiding teams through "
         "architecture decisions and hands-on delivery. Strong Node.js background essential. "
         "You'll coach junior and mid-level engineers and own delivery quality.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE", "JS_TS"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid – client site and remote",
         sen_raw="Lead Developer",
         tech_raw="Node.js, TypeScript, JavaScript, AWS",
         comp_raw="£75,000 – £90,000",
         comp="RANGE_75_99K", cluster="A"),

    make("v13_1_A4", "Staff Software Engineer", "Wise", "London, England, United Kingdom",
         "Wise is looking for a Staff Software Engineer to join our infrastructure team. "
         "At this level you will own entire product areas, set engineering standards, "
         "and partner with product managers and designers. You thrive in ambiguity and "
         "have a proven track record of delivering large-scale technical projects.",
         loc="IN_LONDON", arr="REMOTE", sen="LEVEL_3", tech=["NODE", "AI_ML"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Fully remote — work from anywhere in the UK",
         sen_raw="Staff Software Engineer",
         tech_raw="Node.js, Python, ML pipelines, Kafka",
         comp_raw="£100,000 – £130,000",
         comp="ABOVE_100K", cluster="A"),

    make("v13_1_A5", "Lead Backend Engineer", "GoCardless", "London, England, United Kingdom",
         "GoCardless is growing its payments infrastructure team and we need a Lead Backend "
         "Engineer to own the design and delivery of our real-time payment systems. "
         "You will lead a team of 4 engineers and collaborate closely with product. "
         "You should be comfortable setting coding standards and running technical interviews.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 2 days/week in our London office",
         sen_raw="Lead Backend Engineer",
         tech_raw="Node.js, Ruby, PostgreSQL, Kafka",
         comp_raw="£85,000 – £100,000",
         comp="RANGE_75_99K", cluster="A"),

    # ── Cluster B: Manager titles → LEVEL_3 ────────────────────────────────

    make("v13_1_B1", "Project Manager", "Lloyds Banking Group",
         "London, England, United Kingdom",
         "We are looking for a Project Manager to lead delivery of our digital transformation "
         "initiatives. You will be responsible for managing project timelines, budgets, and "
         "stakeholder communications across multiple workstreams. You'll coordinate with "
         "engineering teams building on Node.js microservices and ensure deliverables are met "
         "on schedule. A proven track record in technical project management is essential.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid working — 3 days in office",
         sen_raw="Project Manager",
         tech_raw="Node.js, Jira, Confluence, AWS project delivery",
         comp_raw="£70,000 – £85,000",
         comp="RANGE_75_99K", cluster="B"),

    make("v13_1_B2", "Project Manager", "Thoughtworks",
         "Manchester, England, United Kingdom",
         "Thoughtworks is hiring a Project Manager to oversee delivery of complex "
         "software consultancy engagements. You will manage client relationships, "
         "sprint ceremonies, risk logs, and resource planning for teams of 8–12 engineers. "
         "You'll work closely with tech leads to unblock delivery and report progress "
         "to senior stakeholders. Experience managing Agile software delivery is required.",
         loc="UK_OTHER", arr="HYBRID", sen="LEVEL_3", tech=["OOS"],
         loc_raw="Manchester, England, United Kingdom",
         arr_raw="Hybrid — office and remote",
         sen_raw="Project Manager",
         tech_raw="Jira, Confluence, Agile delivery, PRINCE2",
         comp_raw="£60,000 – £75,000",
         comp="RANGE_55_74K", cluster="B"),

    make("v13_1_B3", "Product Manager", "Babylon Health",
         "London, England, United Kingdom",
         "We are seeking an experienced Product Manager to own our patient-facing mobile "
         "app experience. You will define the product roadmap, write detailed specs, "
         "and work daily with engineering, design, and clinical teams. You will run "
         "discovery sessions, prioritise the backlog, and measure outcomes using data. "
         "5+ years of product management experience in a consumer tech environment required.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["REACT"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 2 days in our London office",
         sen_raw="Product Manager",
         tech_raw="React Native, Amplitude, SQL, Figma",
         comp_raw="£75,000 – £95,000",
         comp="RANGE_75_99K", cluster="B"),

    make("v13_1_B4", "Product Manager", "Farfetch",
         "London, England, United Kingdom",
         "Farfetch is looking for a Product Manager to join our seller platform team. "
         "You will be the voice of our merchant partners, translating needs into "
         "well-defined features for a team of 6 engineers. You'll own KPIs, run "
         "experimentation, and communicate roadmap progress to the business. "
         "Strong analytical skills and experience with e-commerce platforms required.",
         loc="IN_LONDON", arr="REMOTE", sen="LEVEL_3", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Remote-first — fully flexible",
         sen_raw="Product Manager",
         tech_raw="Node.js microservices, Looker, BigQuery, Amplitude",
         comp_raw="£80,000 – £100,000",
         comp="RANGE_75_99K", cluster="B"),

    make("v13_1_B5", "Technical Manager", "Sky",
         "London, England, United Kingdom",
         "Sky is recruiting a Technical Manager to lead our streaming platform engineering "
         "team. You will be responsible for team performance, technical quality, and "
         "delivery velocity for a squad of 7 engineers. You'll own sprint planning, "
         "1:1s, and technical roadmap in partnership with product leadership. "
         "Strong hands-on Node.js background with 3+ years in management required.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_3", tech=["NODE", "REACT"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — Osterley campus 3 days per week",
         sen_raw="Technical Manager",
         tech_raw="Node.js, React, TypeScript, AWS, Kubernetes",
         comp_raw="£85,000 – £105,000",
         comp="ABOVE_100K", cluster="B"),

    # ── Cluster C: Internship / Roman numeral I → LEVEL_1 ──────────────────

    make("v13_1_C1", "Software Engineer - Internship", "Palantir",
         "London, England, United Kingdom",
         "Palantir is looking for Software Engineer Interns to join our London office "
         "for a 12-week summer programme. You will be embedded in a delivery team, "
         "contributing to real product features alongside experienced engineers. "
         "We're looking for penultimate-year students with strong problem-solving skills "
         "and some experience in JavaScript or Python. No prior professional experience needed.",
         loc="IN_LONDON", arr="IN_OFFICE", sen="LEVEL_1", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="In-office — our London office, 5 days a week during the internship",
         sen_raw="Software Engineer - Internship",
         tech_raw="JavaScript, Python, React, internal tooling",
         comp_raw="£2,500/month",
         comp="BELOW_45K", cluster="C"),

    make("v13_1_C2", "Software Engineer I", "Bloomberg",
         "London, England, United Kingdom",
         "Bloomberg is hiring Software Engineers at the I level (entry-level) to join "
         "our London engineering teams. You will work alongside senior engineers on "
         "data infrastructure and analytics tools. This role is ideal for recent "
         "graduates (0–2 years experience). Strong computer science fundamentals required. "
         "We offer structured mentoring and a clear career progression to SE II and beyond.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_1", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 3 days in our London office",
         sen_raw="Software Engineer I",
         tech_raw="Node.js, C++, Python, internal data systems",
         comp_raw="£45,000 – £55,000",
         comp="RANGE_45_54K", cluster="C"),

    make("v13_1_C3", "Frontend Developer Internship", "Funding Circle",
         "London, England, United Kingdom",
         "Funding Circle is offering a Frontend Developer Internship for students or "
         "recent graduates looking to break into fintech. Over 6 months you will work "
         "with our product team to build and improve customer-facing UI components. "
         "You should have some exposure to React or JavaScript through coursework or "
         "personal projects. This is a paid internship with potential for a graduate offer.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_1", tech=["REACT", "JS_TS"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 2 days in our London office",
         sen_raw="Frontend Developer Internship",
         tech_raw="React, JavaScript, TypeScript, CSS",
         comp_raw="£2,000/month",
         comp="BELOW_45K", cluster="C"),

    make("v13_1_C4", "Backend Engineer I", "Twilio",
         "London, England, United Kingdom",
         "Twilio is growing its London engineering team and is looking for Backend Engineers "
         "at level I — our entry-level engineering track. You will build APIs and "
         "services using Node.js, assisted by a senior engineer buddy throughout your "
         "first 6 months. Requirements: CS degree or equivalent, some exposure to REST "
         "APIs and basic SQL. 0–1 years professional experience expected at this level.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_1", tech=["NODE"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 2 days per week in our London office",
         sen_raw="Backend Engineer I",
         tech_raw="Node.js, PostgreSQL, REST APIs, AWS basics",
         comp_raw="£35,000 – £45,000",
         comp="BELOW_45K", cluster="C"),

    # ── Cluster D: Bare "Full Stack Engineer" → LEVEL_2 ───────────────────

    make("v13_1_D1", "Full Stack Engineer", "Startup (Series A)",
         "London, England, United Kingdom",
         "We're an early-stage startup building workflow automation software and looking "
         "for a Full Stack Engineer to join our small product team. You'll work across "
         "the stack — Node.js backend and React frontend — shipping features independently. "
         "Proven experience with Node.js required. Strong problem-solving skills and "
         "the ability to own features from design to deployment.",
         loc="IN_LONDON", arr="HYBRID", sen="LEVEL_2", tech=["NODE", "REACT"],
         loc_raw="London, England, United Kingdom",
         arr_raw="Hybrid — 2 days in office, rest remote",
         sen_raw="Full Stack Engineer",
         tech_raw="Node.js, React, TypeScript, PostgreSQL",
         comp_raw="£60,000 – £75,000",
         comp="RANGE_55_74K", cluster="D"),

    make("v13_1_D2", "Full Stack Engineer", "TechNinja Solutions",
         "Fully Remote",
         "TechNinja Solutions is seeking a Full Stack Engineer to join our remote team. "
         "You will develop and maintain full-stack web applications using Node.js and "
         "JavaScript/TypeScript. Responsibilities include building RESTful APIs, "
         "collaborating with designers, and conducting code reviews. "
         "Proven experience as a Full Stack Engineer required. Strong problem-solving skills.",
         loc="REMOTE", arr="REMOTE", sen="LEVEL_2", tech=["NODE", "JS_TS"],
         loc_raw="Fully Remote",
         arr_raw="Fully remote — work from anywhere in the UK",
         sen_raw="Full Stack Engineer",
         tech_raw="Node.js, JavaScript, TypeScript, REST APIs",
         comp_raw="£55,000 – £70,000",
         comp="RANGE_55_74K", cluster="D"),

    make("v13_1_D3", "Full Stack Engineer", "Tech Innovators Ltd",
         "Remote (UK)",
         "Tech Innovators Ltd is looking for a Full Stack Engineer to join our dynamic team. "
         "Responsibilities: develop web applications using Node.js, collaborate with "
         "cross-functional teams, write clean maintainable code, troubleshoot applications. "
         "Requirements: proven experience with Node.js, strong problem-solving skills, "
         "excellent communication. Familiarity with cloud services nice to have.",
         loc="REMOTE", arr="REMOTE", sen="LEVEL_2", tech=["NODE"],
         loc_raw="Remote (UK)",
         arr_raw="100% remote — flexible working hours",
         sen_raw="Full Stack Engineer",
         tech_raw="Node.js, Express, MongoDB, AWS",
         comp_raw="£55,000 – £65,000",
         comp="RANGE_55_74K", cluster="D"),

    make("v13_1_D4", "Full Stack Engineer", "Kainos",
         "Belfast, Northern Ireland, United Kingdom",
         "Kainos is a digital services company delivering public sector and commercial "
         "technology. We are looking for a Full Stack Engineer to join our Belfast office. "
         "You will build scalable web applications and APIs using Node.js, contribute "
         "to architecture discussions, and collaborate in Agile teams. 2–4 years of "
         "commercial full-stack experience required.",
         loc="UK_OTHER", arr="HYBRID", sen="LEVEL_2", tech=["NODE", "JS_TS"],
         loc_raw="Belfast, Northern Ireland, United Kingdom",
         arr_raw="Hybrid — Belfast office 3 days per week",
         sen_raw="Full Stack Engineer",
         tech_raw="Node.js, TypeScript, React, PostgreSQL",
         comp_raw="£45,000 – £55,000",
         comp="RANGE_45_54K", cluster="D"),

]

out_path = Path("data/v13_1/contrastive.jsonl")
with open(out_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Written {len(examples)} contrastive examples → {out_path}")

# Summary
from collections import Counter
clusters = Counter(ex["source_file"].split("_")[-1] for ex in examples)
sen_dist = Counter(ex["sen"] for ex in examples)
label_dist = Counter(ex["label"] for ex in examples)
print(f"By cluster: {dict(clusters)}")
print(f"Sen dist:   {dict(sen_dist)}")
print(f"Label dist: {dict(label_dist)}")
