#!/usr/bin/env python3
"""
V12 Phase 1.5: Deterministic JD text preprocessor.

Applied to jd_text BEFORE it reaches the model (at both train AND inference time).
This ensures train/inference consistency — a hard requirement from V10's regression.

What it does:
1. Strip HTML entities (&amp; → &, &nbsp; → space, etc.)
2. Normalize whitespace (collapse multiple newlines/spaces, strip)
3. Remove boilerplate sections (Equal Opportunity, cookie notices)

What it does NOT do (lessons from V11 regression):
- No truncation (leave that to format-for-mlx-v7.ts smart truncation)
- No field-level value changes
- No removal of salary/location clues (regex needs these)
- No aggressive sentence filtering

Usage:
    from preprocess_jd import preprocess_jd
    clean_text = preprocess_jd(raw_jd_text)
"""

import html
import re


# Boilerplate section headers that signal non-relevant content.
# We remove from the header to the next section header or end of text.
BOILERPLATE_HEADERS = [
    r"equal\s+opportunity\s+(?:employer|statement)",
    r"diversity\s+(?:and|&)\s+inclusion",
    r"(?:our\s+)?commitment\s+to\s+(?:diversity|inclusion|equity)",
    r"we\s+are\s+an?\s+equal\s+opportunity",
    r"cookie\s+(?:policy|notice|consent)",
    r"privacy\s+(?:policy|notice|statement)",
    r"data\s+(?:protection|privacy)\s+(?:notice|statement|policy)",
    r"a\s+note\s+on\s+(?:using\s+)?(?:ai|artificial\s+intelligence)",
    r"(?:use\s+of\s+)?artificial\s+intelligence\s+(?:in\s+(?:our|the)\s+)?(?:hiring|recruitment|application)",
]


def preprocess_jd(text: str) -> str:
    """Clean JD text for model consumption.

    This function is deterministic and idempotent — applying it twice
    produces the same result as applying it once.
    """
    if not text:
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Normalize Unicode whitespace
    # Replace non-breaking spaces, zero-width chars, etc.
    text = text.replace("\u00a0", " ")  # &nbsp;
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\u200c", "")   # zero-width non-joiner
    text = text.replace("\u200d", "")   # zero-width joiner
    text = text.replace("\ufeff", "")   # BOM

    # 3. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Remove boilerplate sections
    # These are typically at the end of JDs and don't contain useful signals
    for pattern in BOILERPLATE_HEADERS:
        # Match the header and everything after it until a clear section break
        # or end of text. Use case-insensitive matching.
        boilerplate_re = re.compile(
            rf'(?:^|\n)\s*(?:#{{1,3}}\s*)?(?:{pattern}).*',
            re.IGNORECASE | re.DOTALL
        )
        text = boilerplate_re.sub("", text)

    # 5. Collapse multiple blank lines into max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 7. Strip leading/trailing whitespace per line and overall
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines).strip()

    return text


if __name__ == "__main__":
    # Quick self-test
    sample = """
    We're hiring a &amp; Senior Software Engineer!

    Requirements:
    - Node.js, TypeScript
    - Salary: £80k-£100k

    Equal Opportunity Employer
    We are committed to diversity and inclusion. All qualified applicants
    will receive consideration regardless of race, gender, etc.

    A note on using AI in your job application
    We embrace AI tools but want to see YOUR authentic experience.

    Cookie Policy
    By continuing to use this site you agree to our cookies.
    """
    result = preprocess_jd(sample)
    print("=== INPUT ===")
    print(sample[:200])
    print("\n=== OUTPUT ===")
    print(result)
    print(f"\n=== Stats: {len(sample)} -> {len(result)} chars ({100*len(result)/len(sample):.0f}%) ===")
