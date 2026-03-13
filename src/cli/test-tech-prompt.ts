import * as fs from "node:fs";
import OpenAI from "openai";

async function main() {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const promptPath = process.argv[2] ?? "prompts/teacher_v9.txt";
  const outPath    = process.argv[3] ?? "data/v9/tech_test_new.jsonl";
  const promptTemplate = fs.readFileSync(promptPath, "utf8");
  console.log(`Prompt: ${promptPath}`);
  const jobs = fs.readFileSync("data/v9/tech_prompt_test.jsonl", "utf8")
    .trim().split("\n").map(l => JSON.parse(l));

  const results: unknown[] = [];

  for (const job of jobs) {
    const prompt = promptTemplate
      .replace(/\{\{job_title\}\}/g, job.title || "")
      .replace(/\{\{jd_text\}\}/g, job.jd_text || "");

    const resp = await client.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "Respond with JSON only." },
        { role: "user", content: prompt }
      ],
      max_completion_tokens: 200,
      temperature: 0
    });

    const raw = resp.choices[0]?.message?.content ?? "";
    let parsed: Record<string, unknown> = { tech: ["OOS"], tech_raw: null };
    try { parsed = JSON.parse(raw.trim()); } catch {
      const m = raw.match(/\{[\s\S]*\}/);
      if (m) try { parsed = JSON.parse(m[0]); } catch {}
    }
    results.push({ job_id: job.job_id, title: job.title,
      tech: parsed.tech, tech_raw: parsed.tech_raw });
    process.stdout.write(".");
  }

  fs.writeFileSync(outPath, results.map(r => JSON.stringify(r)).join("\n") + "\n");
  console.log(`\nDone: ${results.length} jobs → ${outPath}`);
}
main().catch(console.error);
