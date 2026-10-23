#!/usr/bin/env python3
"""PostToolUse backstop: flag plan/history leakage in freshly written source.

Fires after every Edit/Write/MultiEdit. Scans the new text for the handful of
tells that recur when an agent lets planning scaffolding or change-history leak
into code comments (see rules/code-comments.md), and, if it finds any, hands a
short reminder back to the agent. Advisory only — the edit has already landed;
the agent decides whether to revise. Anything unexpected exits silently so the
hook can never obstruct editing.
"""
import json
import re
import sys

# Only source files carry code comments. Markdown/TOML/JSON/text are excluded:
# plan files (ANALYSIS_PLAN.md, session notes) legitimately discuss the plan.
CODE_EXT = {
    ".jl", ".py", ".r", ".m", ".js", ".jsx", ".ts", ".tsx",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".rs", ".go", ".java",
}

# (label, compiled pattern). Kept tight to stay high-precision; the first group
# is all but unambiguous, the second catches the prose history-markers the
# user repeatedly has to strike out.
PATTERNS = [
    ("plan/chunk reference", re.compile(r"CHUNK-\d", re.I)),
    # Case-sensitive: the all-caps spellings name the documents, while lowercase
    # "candidates" is ordinary prose.
    ("plan file reference", re.compile(
        r"ANALYSIS_(?:PLAN|SESSION)"
        r"|(?:DESIGN|API|INTEGRATION)_REVIEW_(?:PLAN|SESSION)"
        r"|CANDIDATES|STATE_OF_CODE")),
    # A section number is resolvable only if the document it cuts into is in the
    # repository, so the token before the sign decides and is part of the match.
    # Flagged: a function word or open bracket, which names nothing ("the §12d
    # treatment", "(§3a)"), and an all-caps filename, which names a document that
    # is typically untracked ("CANDIDATES §3a", "DESIGN_NOTES §4"). Left alone: a
    # published standard, which carries its number into the citation ("IEEE 754
    # §5.4", "RFC 8259 §7"), and an author-date reference.
    ("section reference (§ with no in-repo document)",
     re.compile(r"(?:[(\[,]"
                r"|(?<![A-Za-z0-9])[a-z]+[ ]"
                r"|(?<![A-Za-z0-9])[A-Z][A-Z0-9_]{2,}(?:\.\w+)?[ ])[ ]*§\s*\d")),
    ('"as planned"', re.compile(r"\bas planned\b", re.I)),
    ('"Regression:" tag', re.compile(r"\bregression:", re.I)),
    ('history ("Formerly")', re.compile(r"\bformerly\b", re.I)),
    ('history ("Previously")', re.compile(r"\bpreviously\b", re.I)),
    ('history ("used to")', re.compile(r"\bused to\b", re.I)),
    ('hedge ("for now")', re.compile(r"\bfor now\b", re.I)),
]


def new_text(tool_input):
    """Collect the text this tool call introduces, across tool shapes."""
    parts = []
    if "content" in tool_input:                       # Write
        parts.append(tool_input["content"])
    if "new_string" in tool_input:                    # Edit
        parts.append(tool_input["new_string"])
    for edit in tool_input.get("edits", []):          # MultiEdit
        parts.append(edit.get("new_string", ""))
    return "\n".join(parts)


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})
    path = tool_input.get("file_path", "")
    dot = path.rfind(".")
    if dot < 0 or path[dot:].lower() not in CODE_EXT:
        return
    text = new_text(tool_input)
    hits = sorted({label for label, pat in PATTERNS if pat.search(text)})
    if not hits:
        return
    sys.stderr.write(
        "Possible plan/history leakage in comments of {}: {}.\n"
        "Comments should state what is true about the code now, not its history "
        "or the plan it came from (rules/code-comments.md). Re-read the new "
        "comments; revise any that only make sense to someone who watched the "
        "code being written.\n".format(path, ", ".join(hits))
    )
    sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A backstop must never block editing; stay silent on any surprise.
        pass
