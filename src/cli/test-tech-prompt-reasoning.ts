import * as fs from "node:fs";
import OpenAI from "openai";

async function main() {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const promptTemplate = fs.readFileSync("prompts/teacher_tech_v9.txt", "utf8");
  const jobs = fs.readFileSync("data/v9/tech_prompt_test.jsonl", "utf8")
    .trim().split("\n").map(l => JSON.parse(l));

  const model = process.argv[2] ?? "o4-mini";
  console.log(`Model: ${model}, jobs: ${jobs.length}`);

  const results: unknown[] = [];

  for (const job of jobs) {
    const prompt = promptTemplate
      .replace(/\{\{job_title\}\}/g, job.title || "")
      .replace(/\{\{jd_text\}\}/g, job.jd_text || "");

    const resp = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: "Respond with JSON only." },
        { role: "user", content: prompt }
      ],
      max_completion_tokens: 2000,  // reasoning uses tokens internally
    } as Parameters<typeof client.chat.completions.create>[0]);

    const raw = resp.choices[0]?.message?.content ?? "";
    let parsed: Record<string, unknown> = { tech: ["OOS"], tech_raw: null };
    try { parsed = JSON.parse(raw.trim()); } catch {
      const m = raw.match(/\{[\s\S]*\}/);
      if (m) try { parsed = JSON.parse(m[0]); } catch {}
    }
    results.push({
      job_id: job.job_id, title: job.title,
      tech: parsed.tech, tech_raw: parsed.tech_raw
    });
    process.stdout.write(".");
  }

  const outPath = `data/v9/tech_test_${model.replace(/[^a-z0-9]/gi, "_")}.jsonl`;
  fs.writeFileSync(outPath, results.map(r => JSON.stringify(r)).join("\n") + "\n");
  console.log(`\nWritten: ${outPath}`);
}
main().catch(console.error);
