/**
 * Generates highly diverse fake job records using @faker-js/faker.
 * Massive expansion of data pools for titles, locations, tech, and products.
 */

import path from "node:path";
import { faker } from "@faker-js/faker";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type FitLabel = "good_fit" | "maybe" | "bad_fit";

interface GoldenJob {
  job_id: string;
  title: string;
  company: string;
  location?: string;
  jd_text: string;
  label: FitLabel;
  score: number;
  reasoning: string;
}

// ---------------------------------------------------------------------------
// Massive Template Pools
// ---------------------------------------------------------------------------

const SENIOR_TITLES = [
  "Senior Software Engineer",
  "Staff Software Engineer",
  "Lead Backend Engineer",
  "Principal Software Engineer",
  "Senior Platform Engineer",
  "Staff Backend Engineer",
  "Lead Software Engineer",
  "Principal Backend Engineer",
  "Senior Systems Architect",
  "Distinguished Engineer",
  "Lead Full Stack Architect",
  "Senior SRE",
  "VP of Engineering",
  "Founding Engineer (Series A)",
  "Head of Infrastructure",
  "Senior AI Research Engineer",
  "Staff Data Engineer",
  "Senior Cloud Architect",
  "Principal Security Engineer",
  "Lead DevOps Architect",
  "Senior Product Engineer",
];

const MID_TITLES = [
  "Full Stack Engineer",
  "Software Engineer II",
  "Mid-Level Backend Engineer",
  "Full Stack Developer",
  "Software Engineer (L4)",
  "Backend Systems Developer",
  "Application Developer",
  "Systems Engineer",
  "Node.js Developer",
  "Product Developer",
  "Web Scalability Engineer",
  "Integration Engineer",
];

const OTHER_TITLES = [
  "Junior Software Engineer",
  "Graduate Developer",
  "Engineering Manager",
  "Director of Engineering",
  "DevOps Engineer",
  "Cloud Engineer",
  "QA Automation Engineer",
  "Data Engineer",
  "ML Engineer",
  "Security Engineer",
  "Solutions Architect",
  "Technical Product Manager",
  "Scrum Master",
  "Frontend Developer",
  "Mobile Engineer (iOS/Android)",
  "Embedded Systems Engineer",
];

const UK_LOCATIONS = [
  "Remote (UK)",
  "Hybrid - London, UK",
  "Remote (Global)",
  "Hybrid - Manchester, UK",
  "Hybrid - Edinburgh, UK",
  "On-site - London, UK",
  "Hybrid - Bristol, UK",
  "Hybrid - Cambridge, UK",
  "Hybrid - Birmingham, UK",
  "Hybrid - Leeds, UK",
  "On-site - Oxford, UK",
  "On-site - Belfast, UK",
  "Hybrid - Cardiff, UK",
  "On-site - Glasgow, UK",
  "Hybrid - Newcastle, UK",
  "Hybrid - Sheffield, UK",
  "Hybrid - Brighton, UK",
  "Remote (England Only)",
  "Remote (Scotland Only)",
];

const OUTSIDE_UK_LOCATIONS = [
  "Austin, TX, USA",
  "New York, NY, USA",
  "San Francisco, CA, USA",
  "Seattle, WA, USA",
  "Berlin, Germany",
  "Munich, Germany",
  "Amsterdam, Netherlands",
  "Toronto, Canada",
  "Vancouver, Canada",
  "Paris, France",
  "Stockholm, Sweden",
  "Madrid, Spain",
  "Barcelona, Spain",
  "Dublin, Ireland",
  "Sydney, Australia",
  "Singapore",
  "Tokyo, Japan",
  "Helsinki, Finland",
  "Oslo, Norway",
  "Copenhagen, Denmark",
  "Zurich, Switzerland",
  "Dubai, UAE",
  "Lisbon, Portugal",
  "Warsaw, Poland",
  "Prague, Czech Republic",
  "Bangalore, India",
  "Cape Town, South Africa",
];

const TECH_STACKS: { tags: string[]; text: string }[] = [
  {
    tags: ["nodejs", "typescript", "ai"],
    text: "Node.js, TypeScript, and LLM orchestration (LangChain/AutoGPT).",
  },
  {
    tags: ["nodejs", "typescript"],
    text: "Node.js, TypeScript, and PostgreSQL using Prisma/Drizzle.",
  },
  {
    tags: ["nodejs", "javascript"],
    text: "Node.js, Express, and React with complex State Management.",
  },
  {
    tags: ["typescript", "ai"],
    text: "TypeScript, Python, and Vector Databases (Pinecone/Milvus).",
  },
  {
    tags: ["typescript"],
    text: "Next.js, TypeScript, and Tailwind CSS for high-performance frontends.",
  },
  {
    tags: ["python", "ai"],
    text: "Python, PyTorch, and fine-tuning open-source LLMs like Llama 3.",
  },
  {
    tags: ["python"],
    text: "Python, FastAPI, and asynchronous microservices.",
  },
  {
    tags: ["java"],
    text: "Java 21, Spring Boot 3, and Kafka for event-driven architecture.",
  },
  {
    tags: ["go"],
    text: "Golang, gRPC, and Kubernetes-native service development.",
  },
  {
    tags: ["rust"],
    text: "Rust, WASM, and performance-critical systems programming.",
  },
  { tags: ["ruby"], text: "Ruby on Rails 7, Hotwire, and PostgreSQL." },
  {
    tags: ["aws", "devops"],
    text: "AWS (Terraform), Docker, and CircleCI/GitHub Actions.",
  },
  {
    tags: ["graphql"],
    text: "Apollo, GraphQL, and federated schema management.",
  },
];

const PRODUCT_DESCRIPTIONS = [
  "a B2B SaaS platform for enterprise workflow automation",
  "a consumer fintech app revolutionizing micro-investing",
  "an AI-driven suite for automated code reviews",
  "a decentralized finance (DeFi) protocol for institutional grade assets",
  "a healthcare interoperability layer for genomic data",
  "a climate-tech platform tracking carbon offsets in real-time",
  "a high-frequency trading platform for emerging markets",
  "an e-commerce engine powering global luxury brands",
  "a cybersecurity mesh for IoT device management",
  "an ed-tech marketplace for immersive VR learning",
  "a logistics engine optimizing 'last-mile' delivery using ML",
  "a social platform focused on creator-owned digital economies",
];

const RESPONSIBILITIES = [
  "architecting scalable backend systems from scratch",
  "optimizing database queries for tables with billions of rows",
  "mentoring mid-level engineers through code reviews and pair programming",
  "collaborating with stakeholders to define the technical roadmap",
  "building internal tools to improve developer velocity",
  "ensuring 99.99% uptime for mission-critical services",
  "implementing zero-trust security architectures",
  "designing beautiful and intuitive user interfaces",
  "migrating monolithic systems to event-driven microservices",
  "evangelizing engineering best practices across the organization",
];

const BENEFITS = [
  "Unlimited PTO and flexible working hours",
  "Private health insurance and dental coverage",
  "Annual learning and development budget of £2,000",
  "Home office stipend and latest MacBook Pro",
  "Stock options/Equity in a high-growth startup",
  "Generous parental leave and childcare support",
  "Monthly wellness allowance and gym membership",
];

// ---------------------------------------------------------------------------
// Logic & Scoring
// ---------------------------------------------------------------------------

function pickRandom<T>(arr: T[]): T {
  return faker.helpers.arrayElement(arr);
}

const SALARY_OPTIONS: { text: string | null; gbp: number | null }[] = [
  { text: "£120,000 - £140,000", gbp: 130000 },
  { text: "£110,000 - £125,000", gbp: 117500 },
  { text: "£100,000 - £115,000", gbp: 107500 },
  { text: "£90,000 - £105,000", gbp: 97500 },
  { text: "£80,000 - £95,000", gbp: 87500 },
  { text: "£75,000 - £85,000", gbp: 80000 },
  { text: "£65,000 - £75,000", gbp: 70000 },
  { text: "£55,000 - £65,000", gbp: 60000 },
  { text: "£40,000 - £50,000", gbp: 45000 },
  { text: "$120,000 - $150,000", gbp: null },
  { text: null, gbp: null },
];

interface ScoreBreakdown {
  role: number;
  stack: number;
  location: number;
  salary: number;
  total: number;
  label: FitLabel;
}

function scoreRole(title: string): number {
  const t = title.toLowerCase();
  if (
    t.includes("senior") || t.includes("staff") || t.includes("lead") ||
    t.includes("principal") || t.includes("distinguished") || t.includes("vp") ||
    t.includes("head of") || t.includes("founding engineer")
  ) {
    return 25;
  }
  if (
    t.includes("full stack") || t.includes("fullstack") || t.includes("mid") ||
    t.includes("ii") || t.includes("l4") || t.includes("node.js developer")
  ) {
    return 15;
  }
  return 0;
}

function scoreStack(tags: string[]): number {
  let pts = 0;
  if (tags.includes("nodejs")) pts += 10;
  if (tags.includes("typescript") || tags.includes("javascript")) pts += 5;
  if (tags.includes("ai")) pts += 10;
  return Math.min(pts, 25);
}

function scoreLocation(location: string): number {
  const l = location.toLowerCase();
  if (
    l.includes("usa") || l.includes("germany") || l.includes("netherlands") ||
    l.includes("canada") || l.includes("france") || l.includes("sweden") ||
    l.includes("spain") || l.includes("ireland") || l.includes("australia") ||
    l.includes("singapore") || l.includes("japan") || l.includes("finland") ||
    l.includes("norway") || l.includes("denmark") || l.includes("switzerland") ||
    l.includes("israel") || l.includes("uae") || l.includes("portugal") ||
    l.includes("poland") || l.includes("czech") || l.includes("india") ||
    l.includes("south africa")
  ) {
    return -50;
  }
  if (l.includes("remote") || l.includes("london")) return 25;
  if (
    l.includes("uk") || l.includes("manchester") || l.includes("edinburgh") ||
    l.includes("bristol") || l.includes("cambridge") || l.includes("birmingham") ||
    l.includes("leeds") || l.includes("oxford") || l.includes("belfast") ||
    l.includes("cardiff") || l.includes("glasgow") || l.includes("newcastle") ||
    l.includes("sheffield") || l.includes("brighton")
  ) {
    return 10;
  }
  return 0;
}

function scoreSalary(gbp: number | null): number {
  if (gbp === null) return 0;
  if (gbp >= 100000) return 25;
  if (gbp >= 75000) return 15;
  if (gbp >= 55000) return 5;
  if (gbp < 45000) return -30;
  return 0;
}

function computeScore(
  title: string,
  stackTags: string[],
  location: string,
  gbp: number | null,
): ScoreBreakdown {
  const role = scoreRole(title);
  const stack = scoreStack(stackTags);
  const locationPts = scoreLocation(location);
  const salary = scoreSalary(gbp);
  const raw = role + stack + locationPts + salary;
  const total = Math.max(0, Math.min(100, raw));
  const label: FitLabel = total >= 70 ? "good_fit" : total >= 50 ? "maybe" : "bad_fit";
  return { role, stack, location: locationPts, salary, total, label };
}

function buildReasoning(
  breakdown: ScoreBreakdown,
  title: string,
  stackTags: string[],
  location: string,
  salaryText: string | null,
): string {
  const parts: string[] = [];

  if (breakdown.role === 25) parts.push(`${title} is a strong seniority match (+25)`);
  else if (breakdown.role === 15) parts.push(`${title} is an acceptable mid-level role (+15)`);
  else parts.push(`${title} is not a target seniority level (0 pts)`);

  if (stackTags.includes("nodejs")) parts.push("Node.js explicitly required (+10)");
  if (stackTags.includes("typescript") || stackTags.includes("javascript")) parts.push("TS/JS required (+5)");
  if (stackTags.includes("ai")) parts.push("AI/ML experience required (+10)");

  if (breakdown.location === 25) parts.push(`${location} qualifies as remote/London (+25)`);
  else if (breakdown.location === 10) parts.push(`${location} is UK outside London (+10)`);
  else if (breakdown.location === -50) parts.push(`${location} is outside the UK (-50)`);
  else parts.push("Location unknown (0 pts)");

  if (salaryText) {
    if (breakdown.salary === 25) parts.push(`${salaryText} exceeds £100k (+25)`);
    else if (breakdown.salary === 15) parts.push(`${salaryText} is in £75–99k range (+15)`);
    else if (breakdown.salary === 5) parts.push(`${salaryText} is in £55–74k range (+5)`);
    else if (breakdown.salary === -30) parts.push(`${salaryText} is below £45k (-30)`);
  } else {
    parts.push("Salary not disclosed (0 pts)");
  }

  parts.push(`Total: ${breakdown.total}/100`);
  return parts.join(". ").slice(0, 600);
}

function buildJdText(
  company: string,
  title: string,
  stack: string,
  loc: string,
  sal: string | null,
): string {
  const product = pickRandom(PRODUCT_DESCRIPTIONS);
  const resps = faker.helpers.arrayElements(RESPONSIBILITIES, {
    min: 3,
    max: 4,
  });
  const perks = faker.helpers.arrayElements(BENEFITS, { min: 2, max: 3 });

  return `
**About ${company}**
We are an innovative team building ${product}. We're looking for a ${title} who is passionate about building world-class products.

**The Role**
As our ${title}, you will be:
${resps.map((r) => `- ${r}`).join("\n")}

**What You'll Bring**
- Deep expertise in ${stack}
- A track record of solving complex technical hurdles
- Strong opinions, weakly held, regarding architecture and code quality

**Location & Salary**
- Location: ${loc}
- Salary: ${sal ?? "Competitive / Market Rate"}

**Benefits**
${perks.map((p) => `- ${p}`).join("\n")}

How to Apply: Send your GitHub profile and a brief summary of your best work.
  `.trim();
}

// ---------------------------------------------------------------------------
// Main Script
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs();
  const count = getNumberArg(args, "count") ?? 20;
  const seed = getNumberArg(args, "seed") ?? 42;
  const outputPath = getStringArg(args, "output") ?? "data/golden_jobs.jsonl";
  const absoluteOutput = path.resolve(outputPath);

  faker.seed(seed);

  let existing: GoldenJob[] = [];
  try {
    existing = await readJsonlFile<GoldenJob>(absoluteOutput);
  } catch {
    // File doesn't exist yet — that's fine
  }
  const existingIds = new Set(existing.map((j) => j.job_id));

  console.log(`Existing records: ${existing.length}`);
  console.log(`Generating ${count} fake records (seed=${seed})...`);

  const generated: GoldenJob[] = [];
  let attempts = 0;
  const maxAttempts = count * 10;

  while (generated.length < count && attempts < maxAttempts) {
    attempts++;

    const useOutsideUK = faker.number.int({ min: 1, max: 4 }) === 1; // ~25% outside UK
    const tier = faker.number.int({ min: 1, max: 9 });
    const title =
      tier <= 6
        ? pickRandom(SENIOR_TITLES)
        : tier <= 8
          ? pickRandom(MID_TITLES)
          : pickRandom(OTHER_TITLES);

    const company = faker.company.name();
    const location = useOutsideUK
      ? pickRandom(OUTSIDE_UK_LOCATIONS)
      : pickRandom(UK_LOCATIONS);
    const stackChoice = pickRandom(TECH_STACKS);
    const salaryChoice = pickRandom(SALARY_OPTIONS);

    const breakdown = computeScore(title, stackChoice.tags, location, salaryChoice.gbp);
    const jd_text = buildJdText(company, title, stackChoice.text, location, salaryChoice.text);
    const reasoning = buildReasoning(
      breakdown,
      title,
      stackChoice.tags,
      location,
      salaryChoice.text,
    );

    let job_id = faker.string.numeric(10);
    let idAttempts = 0;
    while (existingIds.has(job_id) && idAttempts < 100) {
      job_id = faker.string.numeric(10);
      idAttempts++;
    }
    if (existingIds.has(job_id)) continue;
    existingIds.add(job_id);

    generated.push({ job_id, title, company, location, jd_text, label: breakdown.label, score: breakdown.total, reasoning });
  }

  if (generated.length < count) {
    console.warn(`Warning: only generated ${generated.length}/${count} records after ${maxAttempts} attempts`);
  }

  await writeJsonlFile(absoluteOutput, [...existing, ...generated]);

  const labelCounts = { good_fit: 0, maybe: 0, bad_fit: 0 };
  for (const r of generated) labelCounts[r.label]++;

  console.log(`\nGenerated ${generated.length} records:`);
  console.log(`  good_fit: ${labelCounts.good_fit}`);
  console.log(`  maybe:    ${labelCounts.maybe}`);
  console.log(`  bad_fit:  ${labelCounts.bad_fit}`);
  console.log(`\nTotal in ${outputPath}: ${existing.length + generated.length} records`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
