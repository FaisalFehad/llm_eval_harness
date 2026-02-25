import { readFile, writeFile } from "node:fs/promises";

import { ensureParentDir } from "./paths.js";

export async function readJsonlFile<T>(filePath: string): Promise<T[]> {
  const content = await readFile(filePath, "utf8");
  const lines = content.split(/\r?\n/);
  const items: T[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index] ?? "";
    const line = rawLine.trim();
    if (line.length === 0 || line.startsWith("#")) {
      continue;
    }

    try {
      items.push(JSON.parse(line) as T);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to parse JSONL in ${filePath} at line ${index + 1}: ${message}`,
      );
    }
  }

  return items;
}

export async function writeJsonlFile<T>(filePath: string, items: T[]): Promise<void> {
  await ensureParentDir(filePath);
  const content = items.map((item) => JSON.stringify(item)).join("\n");
  await writeFile(filePath, `${content}\n`, "utf8");
}
