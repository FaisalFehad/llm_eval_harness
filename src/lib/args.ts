export type ParsedArgs = {
  _: string[];
  flags: Record<string, string | boolean>;
};

export function parseArgs(argv: string[] = process.argv.slice(2)): ParsedArgs {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (typeof token !== "string") {
      continue;
    }

    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }

    const withoutPrefix = token.slice(2);
    const equalIndex = withoutPrefix.indexOf("=");

    if (equalIndex >= 0) {
      const key = withoutPrefix.slice(0, equalIndex).trim();
      const value = withoutPrefix.slice(equalIndex + 1).trim();
      if (key.length > 0) {
        flags[key] = value;
      }
      continue;
    }

    const next = argv[i + 1];
    if (typeof next === "string" && !next.startsWith("--")) {
      flags[withoutPrefix] = next;
      i += 1;
      continue;
    }

    flags[withoutPrefix] = true;
  }

  return { _: positional, flags };
}

export function getStringArg(args: ParsedArgs, key: string): string | undefined {
  const value = args.flags[key];
  if (typeof value === "string") {
    return value;
  }

  return undefined;
}

export function getNumberArg(args: ParsedArgs, key: string): number | undefined {
  const value = getStringArg(args, key);
  if (value === undefined || value === "") {
    return undefined;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return undefined;
  }

  return parsed;
}

export function getBooleanArg(args: ParsedArgs, key: string): boolean {
  const value = args.flags[key];
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "off"].includes(normalized)) {
      return false;
    }
  }

  return false;
}
