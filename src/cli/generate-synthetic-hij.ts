/**
 * Generate synthetic training batches H, I, J for V6 student training.
 *
 * Generates and pre-labels 60 jobs covering three gap categories:
 *
 *   BATCH H — Double ambiguity (5 base JDs × 4 variants = 20 jobs)
 *     Base: LONDON_OR_REMOTE + MID_LEVEL (40 pts). Two signals vary:
 *       Node.js presence (NODE_JS_TS=15 vs NONE=0) × salary (RANGE_75_99K=15 vs NO_GBP=0)
 *     Variant A: NODE_JS_TS + RANGE_75_99K → 40+15+15=70 → good_fit
 *     Variant B: NODE_JS_TS + NO_GBP      → 40+15+0=55  → maybe
 *     Variant C: NONE + RANGE_75_99K      → 40+0+15=55  → maybe
 *     Variant D: NONE + NO_GBP            → 40+0+0=40   → bad_fit
 *     Teaching signal: BOTH signals independently shift the label.
 *
 *   BATCH I — Format variance (5 base JDs × 4 formats = 20 jobs)
 *     Same content in 4 different formats (clean bullets / wall of text /
 *     ALL CAPS headers / very short ~80 words). All produce SAME tokens.
 *     Teaching signal: format must not affect classification.
 *
 *   BATCH J — Misleading signals (20 individual jobs, 4 sub-types × 5)
 *     - Type J1: title says "Node.js" but JD says migrating away → tech=NODE
 *                (keyword presence wins — rules say scan for keyword, not context)
 *     - Type J2: "competitive salary" but no £ number → comp=NO_GBP
 *     - Type J3: £ appears in client/funding context, not job salary → comp=NO_GBP
 *     - Type J4: location field says "London" but company section mentions
 *                global HQ in New York / other cities → loc=LONDON_OR_REMOTE
 *                (location field wins over company description)
 *
 * No API needed — JDs are template-generated with controlled signals.
 * Labels computed deterministically from tokens (same logic as semantic-tokens.ts).
 *
 * Usage:
 *   npx tsx src/cli/generate-synthetic-hij.ts \
 *     --output data/v6/batch_hij_labeled.jsonl
 */

import * as fs from "node:fs";
import { parseArgs, getStringArg } from "../lib/args.js";

// ── Score maps (mirrors semantic-tokens.ts, inlined to avoid import) ─────────

const LOC_MAP: Record<string, number> = {
  LONDON_OR_REMOTE: 25, UK_OTHER: 10, OUTSIDE_UK: -50, MISSING: 0,
};
const ROLE_MAP: Record<string, number> = {
  SENIOR_PLUS: 25, MID_LEVEL: 15, NO_SENIORITY: 0,
};
const TECH_MAP: Record<string, number> = {
  NONE: 0, JS_TS: 5, NODE: 10, NODE_JS_TS: 15,
  AI_ML: 10, JS_TS_AI_ML: 15, NODE_AI_ML: 20, NODE_JS_TS_AI_ML: 25,
};
const COMP_MAP: Record<string, number> = {
  NO_GBP: 0, UP_TO_ONLY: 0, BELOW_45K: -30,
  RANGE_55_74K: 5, RANGE_75_99K: 15, ABOVE_100K: 25,
};

function computeScores(loc: string, role: string, tech: string, comp: string) {
  const loc_score = LOC_MAP[loc] ?? 0;
  const role_score = ROLE_MAP[role] ?? 0;
  const tech_score = TECH_MAP[tech] ?? 0;
  const comp_score = COMP_MAP[comp] ?? 0;
  const raw = loc_score + role_score + tech_score + comp_score;
  const score = Math.max(0, Math.min(100, raw));
  const label = score >= 70 ? "good_fit" : score >= 50 ? "maybe" : "bad_fit";
  return { loc_score, role_score, tech_score, comp_score, score, label };
}

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  loc: string;
  role: string;
  tech: string;
  comp: string;
  reasoning: string;
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: string;
  source_file: string;
  augmentation_type: string;
  source_job_id?: string;
  synthetic_target: string;
};

// ── BATCH H — Double Ambiguity ─────────────────────────────────────────────
//
// Base: LONDON_OR_REMOTE (25) + MID_LEVEL (15) = 40 pts
//   Variant A (node+salary): +NODE_JS_TS(15) +RANGE_75_99K(15) = 70 → good_fit
//   Variant B (node+miss):   +NODE_JS_TS(15) +NO_GBP(0)        = 55 → maybe
//   Variant C (nonode+salary):+NONE(0)        +RANGE_75_99K(15) = 55 → maybe
//   Variant D (nonode+miss):  +NONE(0)        +NO_GBP(0)        = 40 → bad_fit
//
// Title must NOT contain Senior/Lead/Staff (to preserve MID_LEVEL role token).
// Location must contain "London" or "Remote" (for LONDON_OR_REMOTE token).

type HBase = {
  id: string;
  company: string;
  title: string;      // No seniority keyword — MID_LEVEL role
  locationStr: string; // Contains London or Remote
  sector: string;
  nodeDesc: string;    // JD body when Node.js is required
  nonodeDesc: string;  // JD body when Python/Go is the stack
  highSalary: string;  // "£85,000 - £95,000 per annum" — RANGE_75_99K
  missSalary: string;  // No £ figure — NO_GBP
};

const H_BASES: HBase[] = [
  {
    id: "h001",
    company: "Paysol Technologies",
    title: "Full Stack Software Engineer",
    locationStr: "London, England, United Kingdom (Remote)",
    sector: "payments infrastructure",
    nodeDesc: `Paysol Technologies is hiring a Full Stack Software Engineer to work on our real-time payments API layer. You will build and maintain services in Node.js and TypeScript, integrating with card networks and banking APIs.

Our stack: Node.js, TypeScript, PostgreSQL, Kafka, Redis, Docker. React on the frontend.

What you need:
- Node.js and TypeScript (3+ years production experience)
- PostgreSQL — schema design and query optimisation
- REST API design and third-party integrations
- Comfortable in a fast-moving fintech environment`,
    nonodeDesc: `Paysol Technologies is hiring a Full Stack Software Engineer to work on our real-time payments infrastructure. You will build and maintain backend services in Python and Go, integrating with card networks and banking APIs.

Our stack: Python 3.12, Go 1.22, PostgreSQL, Kafka, Redis, Docker.

What you need:
- Python and Go (3+ years production experience)
- PostgreSQL — schema design and query optimisation
- REST API design and third-party integrations
- Comfortable in a fast-moving fintech environment`,
    highSalary: "Salary: £85,000 - £95,000 per annum + equity + 25 days holiday + private health",
    missSalary: "Salary: Competitive (full details shared with shortlisted candidates based on experience)",
  },
  {
    id: "h002",
    company: "ShopStream Analytics",
    title: "Software Engineer II",
    locationStr: "London, England, United Kingdom (Hybrid)",
    sector: "e-commerce analytics",
    nodeDesc: `ShopStream Analytics is looking for a Software Engineer II to build data pipeline services for our e-commerce analytics product. You'll write high-throughput ingest services in Node.js and TypeScript connecting retail partners to our analytics engine.

Stack: Node.js, TypeScript, AWS (Lambda, SQS, DynamoDB), Elasticsearch.

Required:
- Node.js and TypeScript (2+ years)
- AWS cloud services experience
- Event-driven architecture patterns
- Good instincts for performance and reliability`,
    nonodeDesc: `ShopStream Analytics is looking for a Software Engineer II to build data pipeline services for our e-commerce analytics product. You'll write high-throughput ingest services in Python connecting retail partners to our analytics engine.

Stack: Python, FastAPI, AWS (Lambda, SQS, DynamoDB), Elasticsearch.

Required:
- Python (2+ years production)
- AWS cloud services experience
- Event-driven architecture patterns
- Good instincts for performance and reliability`,
    highSalary: "Compensation: £85,000 - £95,000 per year + annual bonus + share options",
    missSalary: "Compensation: Market-competitive — we match market rates. Salary discussed at offer stage.",
  },
  {
    id: "h003",
    company: "DataNorth",
    title: "Full Stack Engineer",
    locationStr: "United Kingdom (Remote)",
    sector: "data analytics platforms",
    nodeDesc: `DataNorth is hiring a Full Stack Engineer to build our API layer and streaming data services. You'll work in Node.js and TypeScript daily, creating GraphQL and REST endpoints for our analytics product.

Stack: Node.js, TypeScript, GraphQL, PostgreSQL, GCP.

You'll need:
- Node.js and TypeScript — 2+ years production, you should know streams and async patterns
- GraphQL API design
- PostgreSQL — query optimisation and migrations
- GCP (Cloud Run, Pub/Sub, BigQuery)`,
    nonodeDesc: `DataNorth is hiring a Full Stack Engineer to build our API layer and streaming data services. You'll work in Python and Rust, creating REST endpoints and real-time streaming pipelines for our analytics product.

Stack: Python 3.12, Rust, PostgreSQL, GCP.

You'll need:
- Python (2+ years production) and ideally some Rust experience
- REST API design with FastAPI
- PostgreSQL — query optimisation and migrations
- GCP (Cloud Run, Pub/Sub, BigQuery)`,
    highSalary: "Package: £85,000 - £95,000 base + learning budget + enhanced pension",
    missSalary: "Package: Competitive base — exact figure shared after technical screen.",
  },
  {
    id: "h004",
    company: "GreenOps",
    title: "Mid-Level Backend Developer",
    locationStr: "Remote, United Kingdom",
    sector: "sustainability and carbon accounting software",
    nodeDesc: `GreenOps is growing its engineering team. We need a Mid-Level Backend Developer to build our carbon calculation APIs in Node.js and TypeScript. Our backend is fully Node.js/TypeScript, with PostgreSQL and TimescaleDB for time-series emissions data.

Required skills:
- Node.js and TypeScript (2+ years)
- PostgreSQL and TimescaleDB
- REST API design
- Nice to have: Python for ML model integration, Terraform`,
    nonodeDesc: `GreenOps is growing its engineering team. We need a Mid-Level Backend Developer to build our carbon calculation APIs in Python. Our backend uses Python for computation, with PostgreSQL and TimescaleDB for time-series emissions data.

Required skills:
- Python 3.12 (2+ years production)
- PostgreSQL and TimescaleDB
- REST API design with FastAPI
- Nice to have: Rust for performance modules, Terraform`,
    highSalary: "Salary: £85,000 - £95,000 depending on experience + EMI options + 30 days holiday",
    missSalary: "Salary: Competitive and benchmarked. We aim to pay top-quartile — details on request.",
  },
  {
    id: "h005",
    company: "Claridge Workplace",
    title: "Fullstack Software Engineer",
    locationStr: "London, England, United Kingdom (Hybrid)",
    sector: "B2B workplace management SaaS",
    nodeDesc: `Claridge Workplace is rebuilding its integration layer and needs a Fullstack Software Engineer. You'll design and ship REST and webhook services using Node.js and TypeScript, connecting our SaaS platform to HR systems, calendaring tools, and access control APIs.

Stack: Node.js, TypeScript, Prisma, RabbitMQ, PostgreSQL, Docker.

Experience needed:
- Node.js and TypeScript (2+ years production)
- PostgreSQL and ORM experience (Prisma a plus)
- Event-driven messaging (RabbitMQ or similar)
- Docker and containerised deployments`,
    nonodeDesc: `Claridge Workplace is rebuilding its integration layer and needs a Fullstack Software Engineer. You'll design and ship REST and webhook services using Python and Java, connecting our SaaS platform to HR systems, calendaring tools, and access control APIs.

Stack: FastAPI (Python), Spring Boot (Java), RabbitMQ, PostgreSQL, Docker.

Experience needed:
- Python and Java (2+ years production in either)
- PostgreSQL and ORM experience
- Event-driven messaging (RabbitMQ or similar)
- Docker and containerised deployments`,
    highSalary: "Total compensation: £85,000 - £95,000 base + pension (8% employer) + 25 days holiday",
    missSalary: "Total compensation: Competitive base + pension + holiday. Disclosed post-screening.",
  },
];

function makeHVariant(base: HBase, variantType: "nodehigh" | "nodemiss" | "nohigh" | "nomiss"): LabeledJob {
  const hasNode = variantType === "nodehigh" || variantType === "nodemiss";
  const hasSalary = variantType === "nodehigh" || variantType === "nohigh";

  const techToken = hasNode ? "NODE_JS_TS" : "NONE";
  const compToken = hasSalary ? "RANGE_75_99K" : "NO_GBP";

  const jd_text = `${base.company} — ${base.title}
Location: ${base.locationStr}
Sector: ${base.sector}

${hasNode ? base.nodeDesc : base.nonodeDesc}

What we offer:
${hasSalary ? base.highSalary : base.missSalary}
- Flexible remote working
- Private health insurance
- Professional development budget (£1,500/yr)

${base.company} is an equal opportunity employer.`;

  const techReasoning = hasNode
    ? `tech: Node.js and TypeScript found as required skills → NODE_JS_TS`
    : `tech: no Node.js, no JS/TS found — Python/Go stack → NONE`;
  const compReasoning = hasSalary
    ? `comp: '£85,000 - £95,000' → GBP midpoint £90k → RANGE_75_99K`
    : `comp: 'Competitive salary, details shared' — no GBP annual figure → NO_GBP`;

  const reasoning = `loc: '${base.locationStr}' → London or Remote found → LONDON_OR_REMOTE. role: '${base.title}' → no senior keyword, 'Full Stack'/'Mid-Level'/'Software Engineer II'/'Fullstack' keyword → MID_LEVEL. ${techReasoning}. ${compReasoning}.`;

  const { loc_score, role_score, tech_score, comp_score, score, label } =
    computeScores("LONDON_OR_REMOTE", "MID_LEVEL", techToken, compToken);

  return {
    job_id: `synth_hij_${base.id}_con_${variantType}`,
    title: base.title,
    company: base.company,
    location: base.locationStr,
    jd_text,
    loc: "LONDON_OR_REMOTE",
    role: "MID_LEVEL",
    tech: techToken,
    comp: compToken,
    reasoning,
    loc_score, role_score, tech_score, comp_score, score, label,
    source_file: "synthetic_v5",
    augmentation_type: "contrastive_hij_h",
    source_job_id: `synth_hij_${base.id}_base`,
    synthetic_target: "batch_H_double_ambiguity",
  };
}

// ── BATCH I — Format Variance ──────────────────────────────────────────────
//
// All variants: LONDON_OR_REMOTE(25) + SENIOR_PLUS(25) + NODE_JS_TS(15) + RANGE_75_99K(15) = 80 → good_fit

type IBase = {
  id: string;
  company: string;
  sector: string;
  title: string;
  locationStr: string;
  locDisplay: string;
  tech: string;
  salary: string;
  duties: string[];
  requirements: string[];
};

const I_BASES: IBase[] = [
  {
    id: "i001",
    company: "Habr",
    sector: "property technology",
    title: "Senior Backend Engineer",
    locationStr: "London, England, United Kingdom (Remote)",
    locDisplay: "Remote (UK)",
    tech: "Node.js, TypeScript, PostgreSQL, Redis, AWS",
    salary: "£75,000 - £90,000",
    duties: [
      "Design and build REST and GraphQL APIs serving 2M+ monthly active users",
      "Lead technical design sessions and review code across the backend team",
      "Own the reliability of core property search and listing services",
      "Collaborate with product managers to define and ship new features",
    ],
    requirements: [
      "5+ years backend engineering, at least 2 in Node.js",
      "Strong TypeScript (strict mode) — we use it everywhere",
      "Experience with PostgreSQL and Redis at scale",
      "Comfortable with AWS (Lambda, SQS, RDS, CloudFront)",
    ],
  },
  {
    id: "i002",
    company: "Lexora",
    sector: "legal technology",
    title: "Senior Node.js Developer",
    locationStr: "London, England, United Kingdom (Hybrid)",
    locDisplay: "London (Hybrid)",
    tech: "Node.js, TypeScript, MongoDB, Kafka, GCP",
    salary: "£78,000 - £95,000",
    duties: [
      "Build and maintain the document processing pipeline for our AI-powered contract review product",
      "Design microservices architecture for high-throughput legal document ingestion",
      "Work closely with the ML team to integrate model outputs into the product",
      "Participate in on-call rotation (lightweight, ~2 incidents/month)",
    ],
    requirements: [
      "4+ years Node.js backend development",
      "TypeScript proficiency — strict mode required",
      "MongoDB aggregation pipelines and data modelling",
      "Apache Kafka or similar message queue experience",
    ],
  },
  {
    id: "i003",
    company: "Coverity",
    sector: "insurance technology",
    title: "Senior Software Engineer",
    locationStr: "United Kingdom (Remote)",
    locDisplay: "Fully Remote (UK)",
    tech: "Node.js, TypeScript, DynamoDB, SQS, Terraform",
    salary: "£76,000 - £92,000",
    duties: [
      "Own core insurance pricing and quoting services in Node.js and TypeScript",
      "Build event-driven workflows using AWS SQS and EventBridge",
      "Define and maintain infrastructure-as-code with Terraform",
      "Mentor mid-level engineers and drive engineering standards",
    ],
    requirements: [
      "5+ years backend engineering, Node.js as primary language",
      "TypeScript (strict) required — all new services are TypeScript-first",
      "AWS DynamoDB and SQS experience",
      "Infrastructure mindset — you own your services end to end",
    ],
  },
  {
    id: "i004",
    company: "MedFlow",
    sector: "health technology",
    title: "Senior Backend Developer",
    locationStr: "London, England, United Kingdom (Remote-first)",
    locDisplay: "London / Remote",
    tech: "Node.js, TypeScript, PostgreSQL, RabbitMQ, Docker",
    salary: "£80,000 - £95,000",
    duties: [
      "Build the data integration layer connecting NHS Trusts and private clinics to our platform",
      "Design FHIR-compliant APIs for patient data exchange",
      "Ensure GDPR and Cyber Essentials+ compliance across all data flows",
      "Drive backend architecture decisions in weekly design reviews",
    ],
    requirements: [
      "5+ years backend engineering with Node.js",
      "TypeScript — mandatory, all services are typed end-to-end",
      "PostgreSQL — query optimisation, schema design, migrations",
      "Healthcare or regulated industry experience preferred",
    ],
  },
  {
    id: "i005",
    company: "LearnVault",
    sector: "education technology",
    title: "Senior Platform Engineer",
    locationStr: "London, England, United Kingdom (Hybrid)",
    locDisplay: "London (Hybrid, 2 days/week)",
    tech: "Node.js, TypeScript, Redis, Elasticsearch, Kubernetes",
    salary: "£77,000 - £93,000",
    duties: [
      "Design and operate the content delivery platform serving 500K+ learners",
      "Build real-time progress tracking and adaptive learning APIs in Node.js",
      "Scale our Elasticsearch search layer to support new content types",
      "Drive observability improvements (tracing, metrics, alerting)",
    ],
    requirements: [
      "5+ years Node.js and TypeScript platform engineering",
      "TypeScript strictly typed — experience with Zod or io-ts",
      "Redis caching strategies at scale",
      "Kubernetes operations experience (EKS or GKE)",
    ],
  },
];

function makeIVariant(base: IBase, format: "bullets" | "prose" | "caps" | "short"): LabeledJob {
  let jd_text = "";

  if (format === "bullets") {
    jd_text = `${base.company} — ${base.title}
Location: ${base.locDisplay}
Sector: ${base.sector}

About ${base.company}:
We're a fast-growing ${base.sector} company based in the UK, building products used by hundreds of thousands of customers.

Role Overview:
${base.duties.map((d) => `• ${d}`).join("\n")}

What you'll need:
${base.requirements.map((r) => `• ${r}`).join("\n")}

Compensation:
• ${base.salary} per annum
• 25 days holiday + bank holidays
• Private health insurance
• £1,500 annual learning budget
• Stock options

${base.company} is committed to building a diverse and inclusive team.`;
  } else if (format === "prose") {
    jd_text = `${base.company} is a ${base.sector} company based in the UK. We are looking for a ${base.title} to join our growing engineering team at ${base.locDisplay}. In this role you will be responsible for ${base.duties[0]!.toLowerCase()}, as well as ${base.duties[1]!.toLowerCase()} and ${base.duties[2]!.toLowerCase()}. You will also ${base.duties[3]!.toLowerCase()}. To succeed in this role, you will need ${base.requirements[0]!.toLowerCase()} alongside ${base.requirements[1]!.toLowerCase()}. We also require ${base.requirements[2]!.toLowerCase()} and ${base.requirements[3]!.toLowerCase()}. We offer a salary of ${base.salary} per annum, 25 days annual leave, private medical cover, and a learning and development budget. We are an equal opportunity employer and welcome applicants of all backgrounds. Experience with ${base.tech} is essential for this position.`;
  } else if (format === "caps") {
    jd_text = `${base.company.toUpperCase()} — ${base.title.toUpperCase()}
LOCATION: ${base.locDisplay.toUpperCase()}

COMPANY OVERVIEW
${base.company} is a growing ${base.sector} business serving customers across the UK.

KEY RESPONSIBILITIES
${base.duties.map((d) => `- ${d}`).join("\n")}

REQUIRED SKILLS
${base.requirements.map((r) => `- ${r}`).join("\n")}

WHAT WE OFFER
- SALARY: ${base.salary} PER ANNUM
- 25 DAYS ANNUAL LEAVE
- PRIVATE HEALTH INSURANCE
- LEARNING & DEVELOPMENT BUDGET
- SHARE OPTIONS / EQUITY

${base.company.toUpperCase()} IS AN EQUAL OPPORTUNITY EMPLOYER.`;
  } else {
    // short ~80 words
    jd_text = `${base.company} (${base.sector}) — ${base.title}, ${base.locDisplay}.

Stack: ${base.tech}. Salary: ${base.salary}/yr.

We need a senior engineer with strong Node.js and TypeScript skills to own our core backend services. 5+ years required. PostgreSQL and cloud platform experience expected.

Apply: careers@${base.company.toLowerCase().replace(/\s/g, "")}.com`;
  }

  const reasoning = `loc: '${base.locationStr}' → London or Remote found → LONDON_OR_REMOTE. role: '${base.title}' → 'Senior' keyword → SENIOR_PLUS. tech: Node.js and TypeScript found as required → NODE_JS_TS. comp: '${base.salary}' → GBP range midpoint ~£84k-87k → RANGE_75_99K.`;

  const { loc_score, role_score, tech_score, comp_score, score, label } =
    computeScores("LONDON_OR_REMOTE", "SENIOR_PLUS", "NODE_JS_TS", "RANGE_75_99K");

  return {
    job_id: `synth_hij_${base.id}_con_${format}`,
    title: base.title,
    company: base.company,
    location: base.locationStr,
    jd_text,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning,
    loc_score, role_score, tech_score, comp_score, score, label,
    source_file: "synthetic_v5",
    augmentation_type: "contrastive_hij_i",
    source_job_id: `synth_hij_${base.id}_base`,
    synthetic_target: "batch_I_format_variance",
  };
}

// ── BATCH J — Misleading Signals ──────────────────────────────────────────

type JJob = LabeledJob;

const J_JOBS: Omit<JJob, "loc_score" | "role_score" | "tech_score" | "comp_score" | "score" | "label">[] = [

  // ── J1-J5: Title says "Node.js" but JD says migrating AWAY ──────────────
  // IMPORTANT: tech=NODE — keyword presence wins, not context.
  // Node.js IS mentioned in the JD (in the migration context), so teacher labels it NODE.
  // Teaching signal: don't be fooled by "migrating away" — keyword found = NODE.
  // Score: LONDON_OR_REMOTE(25) + SENIOR_PLUS(25) + NODE(10) + comp = various

  {
    job_id: "synth_hij_j001",
    title: "Senior Node.js Engineer — Platform Migration",
    company: "Orbital Commerce",
    location: "London, England, United Kingdom (Remote)",
    jd_text: `Orbital Commerce is migrating our core platform away from Node.js towards Python (FastAPI + Celery). We are looking for a Senior Engineer with Node.js experience to lead this migration project over the next 12 months.

The ideal candidate understands Node.js deeply enough to read and document our legacy codebase, then port services to Python. Familiarity with Node.js is a hard requirement for reading the existing code — however, all new services will be written in Python.

Responsibilities:
- Lead the incremental migration from Node.js to Python services
- Document and analyse existing Node.js services before porting
- Write Python microservices to replace each Node.js module
- Mentor junior engineers on Python best practices

Salary: £75,000 - £90,000. Remote-first, London office available.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Remote)' → London found → LONDON_OR_REMOTE. role: 'Senior Node.js Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js' keyword found in JD (migration context — familiarity required to read legacy code) → NODE. comp: '£75,000 - £90,000' → midpoint £82.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j1",
    synthetic_target: "batch_J1_migrating_away",
  },
  {
    job_id: "synth_hij_j002",
    title: "Senior Node.js Developer — Greenfield Rebuild",
    company: "Finbourne Technology",
    location: "United Kingdom (Remote)",
    jd_text: `Finbourne Technology is rebuilding our investment data platform. Our existing backend is written in Node.js and we are gradually migrating it to Go. We need a Senior Developer to help drive this migration.

Your day-to-day: reading Node.js services to understand business logic, then rewriting in idiomatic Go. You must be comfortable reading and understanding production Node.js code. Node.js will remain in our codebase for 18+ months.

Target stack: Go 1.23, gRPC, TimescaleDB. But Node.js knowledge is mandatory on day one.

Ideal background:
- Node.js — must be able to read and reason about production Node.js code
- Go (preferred for new code, will upskill if needed)
- Financial services domain knowledge

Salary: £80,000 - £95,000. Fully remote, UK-based.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'United Kingdom (Remote)' → Remote → LONDON_OR_REMOTE. role: 'Senior Node.js Developer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js' keyword found — 'Node.js knowledge is mandatory on day one', 'must be able to read production Node.js code' → NODE. comp: '£80,000 - £95,000' → midpoint £87.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j1",
    synthetic_target: "batch_J1_migrating_away",
  },
  {
    job_id: "synth_hij_j003",
    title: "Senior Full Stack Engineer — Legacy Node.js to Elixir",
    company: "Beam Mobility",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `Beam Mobility is moving our backend from Node.js to Elixir/Phoenix. We are looking for a Senior Full Stack Engineer who can bridge the transition — reading and documenting existing Node.js code while writing new Elixir services.

You must be comfortable working with our Node.js codebase for the first 6 months. All engineers on this project need to be able to read and maintain Node.js during the transition period.

New services will be written in Elixir. The React/TypeScript frontend remains unchanged.

What you'll do:
- Analyse existing Node.js REST APIs for documentation and porting
- Re-implement API surfaces in Elixir/Phoenix
- Maintain the React/TypeScript frontend during the migration
- Write migration tests verifying Node.js and Elixir service equivalence

Salary: £85,000 - £95,000. London hybrid (2 days/wk). 30 days holiday, private health.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London found → LONDON_OR_REMOTE. role: 'Senior Full Stack Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js' keyword found — 'must be comfortable working with our Node.js codebase', 'need to be able to read and maintain Node.js' → NODE. comp: '£85,000 - £95,000' → midpoint £90k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j1",
    synthetic_target: "batch_J1_migrating_away",
  },
  {
    job_id: "synth_hij_j004",
    title: "Senior Backend Engineer — Node.js to Rust Transition",
    company: "Nested",
    location: "London, England, United Kingdom (Remote)",
    jd_text: `Nested is replacing our 6-year-old Node.js monolith with Rust microservices. We need a Senior Backend Engineer to lead this transition, documenting the Node.js system and shipping Rust services in its place.

You will spend the first phase of the role working directly in the Node.js codebase — reading, profiling, and documenting existing services before porting them. Solid Node.js experience is required to succeed in this phase.

The second phase is Rust — we will train you. But you must be able to hit the ground running on the Node.js side from day one.

Skills needed:
- Node.js — must understand the event loop, streams, and our existing service patterns
- Willingness and aptitude to learn Rust
- Distributed systems fundamentals

Compensation: £90,000 - £105,000 per annum, fully remote from anywhere in UK.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE",
    comp: "ABOVE_100K",
    reasoning: "loc: 'London, England, United Kingdom (Remote)' → Remote found → LONDON_OR_REMOTE. role: 'Senior Backend Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js' keyword found — 'solid Node.js experience is required', 'hit the ground running on Node.js from day one' → NODE. comp: '£90,000 - £105,000' → midpoint £97.5k → RANGE_75_99K. Wait — midpoint £97.5k is below £100k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j1",
    synthetic_target: "batch_J1_migrating_away",
  },
  {
    job_id: "synth_hij_j005",
    title: "Senior Software Engineer — Node.js to Java Migration",
    company: "Vouchercloud",
    location: "Bristol, England, United Kingdom (Hybrid)",
    jd_text: `Vouchercloud's engineering team is rewriting our Node.js services in Java (Spring Boot). We need a Senior Software Engineer who can read and document our existing Node.js codebase, then faithfully rewrite services in Java.

You must have working knowledge of Node.js — this is non-negotiable for the first phase of the project. You don't need to write new Node.js code, but you must be able to understand and explain existing Node.js service behaviour to the Java team.

You'll be working with: Node.js (reading only), Java 21, Spring Boot 3, PostgreSQL, RabbitMQ, Docker.

Pay: £75,000 - £88,000 per annum. Bristol hybrid (2 days/wk). Equal opportunities employer.`,
    loc: "UK_OTHER",
    role: "SENIOR_PLUS",
    tech: "NODE",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'Bristol, England, United Kingdom (Hybrid)' → Bristol found, not London/Remote → UK_OTHER. role: 'Senior Software Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js' keyword found — 'working knowledge of Node.js is non-negotiable', 'must be able to understand existing Node.js service behaviour' → NODE. comp: '£75,000 - £88,000' → midpoint £81.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j1",
    synthetic_target: "batch_J1_migrating_away",
  },

  // ── J6-J10: "Competitive salary" — no £ number visible ──────────────────
  // loc=LONDON_OR_REMOTE, role=SENIOR_PLUS, tech=NODE_JS_TS, comp=NO_GBP → 65 → maybe
  {
    job_id: "synth_hij_j006",
    title: "Senior Node.js Engineer",
    company: "Curve",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `Curve is a fintech superapp. We're looking for a Senior Node.js Engineer to join our payments platform team.

You'll work on our transaction processing engine, fraud detection hooks, and card management APIs — all in Node.js and TypeScript.

Stack: Node.js, TypeScript, PostgreSQL, Kafka, Redis, AWS.

Requirements:
- 5+ years Node.js production experience
- TypeScript (strict mode, no any)
- PostgreSQL — migrations, query optimisation, scaling
- Event-driven architecture (Kafka or similar)

What we offer:
- Competitive salary commensurate with experience
- Share options in a well-funded fintech
- Remote-flexible hybrid (2 days/wk London)
- 25 days holiday, private medical (Bupa), pension (5% matched)

Curve is an equal opportunity employer.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London found → LONDON_OR_REMOTE. role: 'Senior Node.js Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js and TypeScript' found as required → NODE_JS_TS. comp: 'Competitive salary commensurate with experience' → no GBP annual salary figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j2",
    synthetic_target: "batch_J2_competitive_salary",
  },
  {
    job_id: "synth_hij_j007",
    title: "Senior Software Engineer (Node.js)",
    company: "Tractable",
    location: "United Kingdom (Remote)",
    jd_text: `Tractable uses computer vision and AI to automate insurance claims. We are looking for a Senior Software Engineer specialising in Node.js and TypeScript.

You will build and maintain our AI model serving layer and the APIs that connect insurers and repairers to our platform.

What you need:
• Node.js (4+ years production, event loop, streams)
• TypeScript — we use it strictly
• REST and gRPC API design experience
• AWS — Lambda, ECS, API Gateway

Compensation:
We offer a market-leading, experience-based compensation package. Exact salary is discussed and agreed during the offer stage to ensure fairness across the team.

Other benefits: unlimited PTO, stock options, home office budget £500, learning budget £2,000/yr.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'United Kingdom (Remote)' → Remote → LONDON_OR_REMOTE. role: 'Senior Software Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: 'salary discussed and agreed during offer stage' — no GBP annual figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j2",
    synthetic_target: "batch_J2_competitive_salary",
  },
  {
    job_id: "synth_hij_j008",
    title: "Senior Backend Engineer — Node.js / TypeScript",
    company: "Multiverse",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `Multiverse is an apprenticeship platform helping people build skills at leading companies. We're hiring a Senior Backend Engineer to work on our coaching and curriculum delivery APIs.

Your stack: Node.js (primary), TypeScript, GraphQL, PostgreSQL, Datadog.

Key experience needed:
- Node.js and TypeScript — must be your daily language
- GraphQL API design and federated schema experience
- PostgreSQL at scale
- Strong CS fundamentals and systems thinking

Pay and benefits: Competitive salary, reviewed annually against market data. Comprehensive benefits including Vitality health insurance, enhanced parental leave, and equity. Exact base salary shared at first call.

Multiverse is an equal opportunities employer.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London → LONDON_OR_REMOTE. role: 'Senior Backend Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript as daily language → NODE_JS_TS. comp: 'Competitive salary' and 'exact base salary shared at first call' — no GBP figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j2",
    synthetic_target: "batch_J2_competitive_salary",
  },
  {
    job_id: "synth_hij_j009",
    title: "Senior Node.js Developer",
    company: "Octopus Energy",
    location: "London, England, United Kingdom (Remote-first)",
    jd_text: `Octopus Energy is reinventing the energy market. Our Kraken platform serves 50M+ customers globally. We are hiring Senior Node.js Developers to work on UK customer experience APIs.

What you'll work on:
- Real-time smart meter data ingestion APIs in Node.js and TypeScript
- Customer billing and tariff calculation services
- A/B test infrastructure for energy tariff experiments

Stack: Node.js, TypeScript, Django (Python) for some services, PostgreSQL, Kubernetes.

We are transparent about most things but not salary publicly — we believe in fair pay and make offers based on your experience in a conversation. What we can say: you will be paid well and at the 75th-percentile for the level.

Benefits: 30 days holiday, company shares, pension (5% matched), home energy discount.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Remote-first)' → London/Remote → LONDON_OR_REMOTE. role: 'Senior Node.js Developer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: 'we do not disclose salary publicly, offers based on experience' — no GBP figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j2",
    synthetic_target: "batch_J2_competitive_salary",
  },
  {
    job_id: "synth_hij_j010",
    title: "Senior Engineer — Node.js Platform",
    company: "Bought By Many",
    location: "United Kingdom (Remote)",
    jd_text: `Bought By Many (ManyPets) is a leading pet insurance brand. We're hiring a Senior Engineer for our policy platform team, working in Node.js and TypeScript.

You'll own our insurance quoting and binding APIs, integrated with multiple underwriter systems.

Required:
- Node.js + TypeScript (4+ years production)
- RESTful and event-driven API design
- Familiarity with insurance or financial services workflows preferred
- AWS experience (SQS, RDS, Lambda)

We don't post salaries publicly. Our compensation is personalised based on skills, experience, and market data. We are a Living Wage employer and can confirm the salary is comfortably above the market average for this level.

Benefits: 30 days holiday, pet insurance, private medical, equity. Remote-first, UK residents only.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'United Kingdom (Remote)' → Remote → LONDON_OR_REMOTE. role: 'Senior Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js + TypeScript required → NODE_JS_TS. comp: 'We don't post salaries publicly' — no GBP annual figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j2",
    synthetic_target: "batch_J2_competitive_salary",
  },

  // ── J11-J15: £ appears in client/funding context, NOT as job salary ──────
  // loc=LONDON_OR_REMOTE, role=SENIOR_PLUS, tech=NODE_JS_TS, comp=NO_GBP → 65 → maybe
  {
    job_id: "synth_hij_j011",
    title: "Senior Node.js Engineer",
    company: "Paddle",
    location: "London, England, United Kingdom (Remote-first)",
    jd_text: `Paddle is the merchant of record for software companies. We process £3B+ in payments annually on behalf of our customers. Our engineering team is growing following our £132M Series D.

We need a Senior Node.js Engineer to join our payment routing team, building the APIs that move money for thousands of SaaS businesses.

Stack: Node.js, TypeScript, PostgreSQL, Kafka, AWS.

Requirements:
- Node.js and TypeScript (5+ years production)
- PostgreSQL — complex queries, partitioning, replication
- Payment systems or fintech background preferred

Salary: Not disclosed publicly. Competitive package discussed during screening. Includes equity, private health, 25 days holiday.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Remote-first)' → London → LONDON_OR_REMOTE. role: 'Senior Node.js Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£3B+ in payments' and '£132M Series D' are company metrics not job salary; 'not disclosed publicly' → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j3",
    synthetic_target: "batch_J3_pounds_in_context",
  },
  {
    job_id: "synth_hij_j012",
    title: "Senior Backend Engineer (Node.js)",
    company: "Cleo",
    location: "United Kingdom (Remote)",
    jd_text: `Cleo is a financial wellbeing app. We've raised over £80M to date and our users save an average of £350/month using Cleo. We're hiring a Senior Backend Engineer to join our core banking integration team.

You'll build the services that connect Cleo to 1,000+ banks via Open Banking APIs.

What you need:
- Node.js and TypeScript — our whole backend is Node.js
- REST APIs and OAuth 2.0 (Open Banking is standards-based)
- PostgreSQL and Redis

Total comp: equity-heavy package. We do pay well but ask that you come in for a chat before we make an offer to ensure alignment.

We're fully remote, UK-based only.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'United Kingdom (Remote)' → Remote → LONDON_OR_REMOTE. role: 'Senior Backend Engineer' → 'Senior' → SENIOR_PLUS. tech: 'Node.js and TypeScript' — whole backend → NODE_JS_TS. comp: '£80M raised' and '£350/month savings' are company/product metrics not job salary; job comp 'not publicly listed' → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j3",
    synthetic_target: "batch_J3_pounds_in_context",
  },
  {
    job_id: "synth_hij_j013",
    title: "Senior Software Engineer — Node.js",
    company: "Farewill",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `Farewill has written over £5B worth of wills since founding. We help people protect their families and leave meaningful legacies.

We're looking for a Senior Software Engineer to work on our core platform — will creation, user accounts, and partner integrations.

Stack: Node.js, TypeScript, PostgreSQL, Stripe, AWS Lambda.

Required skills:
- Node.js and TypeScript (production experience, 4+ years)
- PostgreSQL — schema design, migrations, data integrity
- API design and third-party integrations
- Thoughtful about technical debt and code quality

What we offer: competitive salary (benchmark-based, discussed in offer stage), meaningful equity, private health insurance, 28 days holiday. Hybrid working from our London office 2 days per week.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London → LONDON_OR_REMOTE. role: 'Senior Software Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£5B worth of wills' is company metric; salary 'competitive, discussed at offer stage' — no GBP figure → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j3",
    synthetic_target: "batch_J3_pounds_in_context",
  },
  {
    job_id: "synth_hij_j014",
    title: "Senior Node.js Developer",
    company: "Wise",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `Wise (formerly TransferWise) moves £10B+ per month across borders for 16M+ customers. We are hiring Senior Node.js Developers for our back-end platform teams.

You'll be part of the team building payment rails that handle billions of pounds in transfers every month.

Requirements:
- Node.js — 4+ years production, you should know the event loop, streams, and cluster
- TypeScript — strict mode, advanced generics welcomed
- High-scale systems (10k+ RPS, multi-region)

Compensation policy: Wise pays to the 75th percentile of the market in each location. We do not disclose the exact band publicly. Your offer will be benchmarked during the hiring process.

Benefits: 25 days holiday, private health (AXA), RSUs, flexible hours.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London → LONDON_OR_REMOTE. role: 'Senior Node.js Developer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£10B+ per month' is transaction volume not job salary; 'do not disclose band publicly' → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j3",
    synthetic_target: "batch_J3_pounds_in_context",
  },
  {
    job_id: "synth_hij_j015",
    title: "Senior Engineer — Node.js API Platform",
    company: "Growth Street",
    location: "United Kingdom (Remote)",
    jd_text: `Growth Street provides revenue-based financing to UK SMEs, having deployed over £200M in loans to date. Our engineering team is small and high-impact — every engineer owns end-to-end features.

We're hiring a Senior Engineer to own our loan origination and risk APIs.

Stack: Node.js, TypeScript, NestJS, PostgreSQL, AWS.

What we're looking for:
- Node.js and TypeScript expertise (NestJS a bonus)
- PostgreSQL: complex financial queries, audit trails, data integrity
- Comfort with financial data and regulatory constraints

Compensation: growth-stage startup, so we offer equity (real upside) alongside a market salary. We don't list the exact figure here — it is discussed during the process and calibrated to your experience.

Remote, UK residents only.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "NO_GBP",
    reasoning: "loc: 'United Kingdom (Remote)' → Remote → LONDON_OR_REMOTE. role: 'Senior Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript expertise required → NODE_JS_TS. comp: '£200M in loans' is company metric; salary 'not listed here, discussed during process' → NO_GBP.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j3",
    synthetic_target: "batch_J3_pounds_in_context",
  },

  // ── J16-J20: Location field says "London" — red herring is company global HQ ──
  // IMPORTANT: loc=LONDON_OR_REMOTE — location field wins over company description.
  // Teaching signal: seeing "New York headquarters" or "global offices" in company
  // description should NOT override the location field which says London.
  // Score: LONDON_OR_REMOTE(25) + SENIOR_PLUS(25) + NODE_JS_TS(15) + RANGE_75_99K(15) = 80 → good_fit
  {
    job_id: "synth_hij_j016",
    title: "Senior Node.js Engineer",
    company: "Contentful",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `About Contentful: Contentful is a content platform headquartered in Berlin, with major engineering offices in New York, San Francisco, and London. We serve 30B+ API requests per month across our global infrastructure.

This role is based in our London office.

We are looking for a Senior Node.js Engineer for our content delivery team.

Stack: Node.js, TypeScript, Kafka, Elasticsearch, AWS.

Requirements:
- Node.js and TypeScript — 5+ years production
- High-throughput API design (>10k RPS)
- Kafka or similar event streaming experience

Working from our London Shoreditch office (hybrid, 3 days/wk). No relocation required.

Salary: £85,000 - £95,000 + stock options + 25 days holiday + private health.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London found in location field → LONDON_OR_REMOTE (company's Berlin HQ and global offices in description do not override the location field). role: 'Senior Node.js Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£85,000 - £95,000' → midpoint £90k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j4",
    synthetic_target: "batch_J4_london_field_wins",
  },
  {
    job_id: "synth_hij_j017",
    title: "Senior Backend Engineer — Node.js",
    company: "Brex",
    location: "London, England, United Kingdom (Remote)",
    jd_text: `About Brex: Brex is a US-headquartered fintech company (San Francisco, New York) that provides business banking and spend management to startups and enterprises. We have European teams based in London and Dublin, serving our growing international customer base.

This position is for our London-based engineering team (remote).

We are hiring a Senior Backend Engineer to work on our European payment rails in Node.js and TypeScript.

Stack: Node.js, TypeScript, PostgreSQL, Apache Kafka, GCP.

What you need:
- Node.js and TypeScript (5+ years)
- PostgreSQL at scale
- Kafka for event-driven architecture
- Payments or fintech experience preferred

Salary: £80,000 - £95,000 per annum + equity + 25 days + health cover.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Remote)' → London found in location field → LONDON_OR_REMOTE (company's San Francisco/New York HQ in description does not override location field). role: 'Senior Backend Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£80,000 - £95,000' → midpoint £87.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j4",
    synthetic_target: "batch_J4_london_field_wins",
  },
  {
    job_id: "synth_hij_j018",
    title: "Senior Node.js Developer",
    company: "Stripe",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `About Stripe: Stripe is a global technology company headquartered in South San Francisco and Dublin, Ireland. We have offices in New York, Singapore, Tokyo, and London. Stripe's mission is to increase the GDP of the internet.

This role is in our London engineering hub.

We are hiring a Senior Node.js Developer to work on our UK and European merchant integrations team.

Requirements:
- Node.js and TypeScript — deep expertise, 5+ years
- REST API design at scale
- PostgreSQL and distributed systems
- Payments domain knowledge preferred

The role is based in London, with occasional travel to Dublin (1-2x per year).

Salary: £85,000 - £95,000 base + RSUs + annual bonus + comprehensive benefits. London salary scale applies.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London found in location field → LONDON_OR_REMOTE (company mentions San Francisco, Dublin, New York, Singapore, Tokyo in description — these do not override location field). role: 'Senior Node.js Developer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£85,000 - £95,000 base' → midpoint £90k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j4",
    synthetic_target: "batch_J4_london_field_wins",
  },
  {
    job_id: "synth_hij_j019",
    title: "Senior Software Engineer — Node.js",
    company: "Paysafe",
    location: "London, England, United Kingdom (Remote)",
    jd_text: `About Paysafe: Paysafe is a global payments platform with its registered corporate address in London, UK. Our engineering centres are distributed globally across Sofia, Toronto, and Hyderabad, with commercial teams in London and New York.

This role is part of our London-registered entity and the candidate will be UK-based (remote).

We need a Senior Software Engineer for the UK payments routing team.

Tech stack: Node.js, TypeScript, Azure, SQL Server.

What we need:
- Node.js and TypeScript (5+ years)
- Enterprise-scale databases (SQL Server or PostgreSQL)
- Azure cloud services experience
- Regulated payment environment experience

Compensation: £80,000 - £95,000 per annum. UK remote. Bonus + pension + private health.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Remote)' → London/Remote found in location field → LONDON_OR_REMOTE (engineering centres in Sofia, Toronto, Hyderabad are in description — do not override location field). role: 'Senior Software Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£80,000 - £95,000' → midpoint £87.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j4",
    synthetic_target: "batch_J4_london_field_wins",
  },
  {
    job_id: "synth_hij_j020",
    title: "Senior Node.js Engineer — API Platform",
    company: "Fenergo",
    location: "London, England, United Kingdom (Hybrid)",
    jd_text: `About Fenergo: Fenergo provides client lifecycle management software for financial institutions. Our engineering hub is in Dublin, Ireland, with commercial teams in our London, New York, Singapore, and Sydney offices. The majority of our engineers are based in Dublin.

This role is part of our London commercial engineering team, working on UK-specific regulatory workflows.

We are hiring a Senior Node.js Engineer for our London team.

Stack: Node.js, TypeScript, Azure Cosmos DB, Azure Service Bus.

Requirements:
- Node.js and TypeScript — 4+ years production
- Azure cloud services (Cosmos DB, Service Bus)
- Financial services regulatory experience (KYC, AML) preferred

This role is London-based hybrid (2 days/wk in our London office).

Salary: £85,000 - £98,000 + annual bonus + stock. Private health, pension, 25 days PTO.`,
    loc: "LONDON_OR_REMOTE",
    role: "SENIOR_PLUS",
    tech: "NODE_JS_TS",
    comp: "RANGE_75_99K",
    reasoning: "loc: 'London, England, United Kingdom (Hybrid)' → London found in location field → LONDON_OR_REMOTE (company description mentions Dublin engineering hub, New York, Singapore — these do not override location field which is London). role: 'Senior Node.js Engineer' → 'Senior' → SENIOR_PLUS. tech: Node.js and TypeScript required → NODE_JS_TS. comp: '£85,000 - £98,000' → midpoint £91.5k → RANGE_75_99K.",
    source_file: "synthetic_v5",
    augmentation_type: "misleading_hij_j4",
    synthetic_target: "batch_J4_london_field_wins",
  },
];

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const outputPath = getStringArg(args, "output") ?? "data/v6/batch_hij_labeled.jsonl";

  const allJobs: LabeledJob[] = [];

  // ── Batch H ──────────────────────────────────────────────────────────────
  console.log("Generating Batch H (double ambiguity)...");
  for (const base of H_BASES) {
    for (const variant of ["nodehigh", "nodemiss", "nohigh", "nomiss"] as const) {
      allJobs.push(makeHVariant(base, variant));
    }
  }
  console.log(`  Generated ${H_BASES.length * 4} Batch H jobs`);

  // ── Batch I ──────────────────────────────────────────────────────────────
  console.log("Generating Batch I (format variance)...");
  for (const base of I_BASES) {
    for (const fmt of ["bullets", "prose", "caps", "short"] as const) {
      allJobs.push(makeIVariant(base, fmt));
    }
  }
  console.log(`  Generated ${I_BASES.length * 4} Batch I jobs`);

  // ── Batch J ──────────────────────────────────────────────────────────────
  console.log("Generating Batch J (misleading signals)...");
  for (const rawJob of J_JOBS) {
    const { loc_score, role_score, tech_score, comp_score, score, label } =
      computeScores(rawJob.loc, rawJob.role, rawJob.tech, rawJob.comp);
    allJobs.push({ ...rawJob, loc_score, role_score, tech_score, comp_score, score, label });
  }
  console.log(`  Generated ${J_JOBS.length} Batch J jobs`);

  // ── Fix j004: comp is RANGE_75_99K not ABOVE_100K (midpoint £97.5k < £100k) ─
  // (already set correctly in J_JOBS above — just verify)

  // ── Print distribution ────────────────────────────────────────────────────
  const counts: Record<string, number> = { good_fit: 0, maybe: 0, bad_fit: 0 };
  for (const j of allJobs) counts[j.label]!++;

  console.log("\n" + "═".repeat(60));
  console.log("BATCH H/I/J GENERATION COMPLETE");
  console.log("═".repeat(60));
  console.log(`Total jobs generated: ${allJobs.length}`);
  console.log(`\nLabel distribution:`);
  for (const [lbl, cnt] of Object.entries(counts)) {
    console.log(`  ${lbl.padEnd(12)}: ${cnt}`);
  }

  // ── Verify H variants ────────────────────────────────────────────────────
  console.log("\nBatch H verification (base = LONDON_OR_REMOTE + MID_LEVEL = 40 pts):");
  const hJobs = allJobs.filter((j) => j.job_id.includes("_h0"));
  for (const j of hJobs) {
    const ok = j.loc === "LONDON_OR_REMOTE" && j.role === "MID_LEVEL";
    console.log(`  ${j.job_id}: loc=${j.loc}, role=${j.role}, tech=${j.tech}, comp=${j.comp}, score=${j.score}, label=${j.label} ${ok ? "✓" : "⚠ WRONG"}`);
  }

  // ── Verify I variants ────────────────────────────────────────────────────
  console.log("\nBatch I verification (all should be score=80, good_fit):");
  const iJobs = allJobs.filter((j) => j.job_id.includes("_i0"));
  for (const j of iJobs) {
    const ok = j.score === 80 && j.label === "good_fit";
    console.log(`  ${j.job_id}: score=${j.score}, label=${j.label} ${ok ? "✓" : "⚠ WRONG"}`);
  }

  // ── Verify J variants ────────────────────────────────────────────────────
  const jJobs = allJobs.filter((j) => j.job_id.startsWith("synth_hij_j"));
  const j1 = jJobs.filter((j) => j.synthetic_target.includes("J1"));
  const j2 = jJobs.filter((j) => j.synthetic_target.includes("J2"));
  const j3 = jJobs.filter((j) => j.synthetic_target.includes("J3"));
  const j4 = jJobs.filter((j) => j.synthetic_target.includes("J4"));

  console.log(`\nBatch J verification:`);
  const techsJ1 = [...new Set(j1.map((j) => j.tech))];
  console.log(`  J1 (migrating away): tech=${techsJ1} (expected: NODE) ${techsJ1.every((t) => t === "NODE") ? "✓" : "⚠ WRONG"}`);
  const compsJ2 = [...new Set(j2.map((j) => j.comp))];
  console.log(`  J2 (competitive salary): comp=${compsJ2} (expected: NO_GBP) ${compsJ2.every((c) => c === "NO_GBP") ? "✓" : "⚠ WRONG"}`);
  const compsJ3 = [...new Set(j3.map((j) => j.comp))];
  console.log(`  J3 (£ in context): comp=${compsJ3} (expected: NO_GBP) ${compsJ3.every((c) => c === "NO_GBP") ? "✓" : "⚠ WRONG"}`);
  const locsJ4 = [...new Set(j4.map((j) => j.loc))];
  console.log(`  J4 (London field wins): loc=${locsJ4} (expected: LONDON_OR_REMOTE) ${locsJ4.every((l) => l === "LONDON_OR_REMOTE") ? "✓" : "⚠ WRONG"}`);

  // ── Write output ──────────────────────────────────────────────────────────
  const dir = outputPath.split("/").slice(0, -1).join("/");
  if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const outStream = fs.createWriteStream(outputPath);
  for (const job of allJobs) {
    outStream.write(JSON.stringify(job) + "\n");
  }
  outStream.end();

  console.log(`\nOutput: ${outputPath} (${allJobs.length} jobs)`);
  console.log("\nNext steps (V6 trains on NEW data only — no V5 mixing):");
  console.log("  1. npx tsx src/cli/format-for-mlx.ts \\");
  console.log("       --input data/v6/batch_hij_labeled.jsonl \\");
  console.log("       --output-dir data/v6/mlx \\");
  console.log("       --prompt prompts/student_v5.txt \\");
  console.log("       --valid-pct 10");
  console.log("  2. python -m mlx_lm.lora \\");
  console.log("       --config finetune/lora_config_v6.yaml \\");
  console.log("       --resume-adapter-file finetune/adapters_v5.1/0000875_adapters.safetensors");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
