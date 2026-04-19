/**
 * Prompt wrapper — reads a plain-text prompt file and wraps it in chat format.
 *
 * Usage in promptfooconfig.yaml (from configs/ directory):
 *   prompts:
 *     - id: file://wrap_prompt.cjs:v15_production
 *       label: "v15-production"
 *
 * File caching: prompt files are read once and cached for the duration of the eval run.
 */
const fs = require("fs");
const path = require("path");

const _cache = {};

function makePrompt(filename, label) {
  const filepath = path.join(__dirname, "..", filename);
  const fn = function ({ vars }) {
    if (!_cache[filename]) {
      _cache[filename] = fs.readFileSync(filepath, "utf-8");
    }
    let text = _cache[filename];

    // Substitute Nunjucks-style {{var}} placeholders with test case values
    for (const [key, value] of Object.entries(vars)) {
      text = text.replace(
        new RegExp(`\\{\\{\\s*${key}\\s*\\}\\}`, "g"),
        value ?? ""
      );
    }

    return [
      { role: "system", content: "Respond with JSON only." },
      { role: "user", content: text },
    ];
  };
  // Set function name and label for promptfoo UI display
  Object.defineProperty(fn, "name", { value: label });
  fn.label = label;
  return fn;
}

// Export one function per prompt variant
// __dirname = versions/v15/configs → "../prompts/" resolves to versions/v15/prompts/
module.exports = {
  v15_production: makePrompt("prompts/student.production.txt", "v15-production"),
  v15_fix4: makePrompt("prompts/student.fix4.txt", "v15-fix4"),
  v15_fix3: makePrompt("prompts/student.fix3.txt", "v15-fix3"),
  v15_fix2: makePrompt("prompts/student.fix2.txt", "v15-fix2"),
  v15_fix1: makePrompt("prompts/student.fix1.txt", "v15-fix1"),
  v15_fix5: makePrompt("prompts/student.fix5.txt", "v15-fix5"),
  v15_base: makePrompt("prompts/student.base.txt", "v15-base"),
};
