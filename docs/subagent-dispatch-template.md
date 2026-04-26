# Subagent Dispatch Template

Every `Agent` tool call in this project must prepend the **SCOPE** block below to its prompt, with all `{{...}}` placeholders filled in. The controller (main thread) is responsible for this — never dispatch a subagent without it.

If a subagent ignores the scope block, that is a controller failure: the prompt did not enforce the rules clearly enough. Tighten it before retrying.

---

## Template

```
## SCOPE — read this before doing anything else

You are working on the **Bayan-Graph** project. The full guardrails are in `CLAUDE.md` at the repo root — read it now if you have not already.

**Working directory:** {{absolute path to worktree, e.g. C:\Users\Owner\Desktop\Projects\Bayan-Graph\.worktrees\plan-01-foundation}}
**Branch:** {{branch name, e.g. feat/plan-01-foundation}}
**Active plan:** {{path to plan file, relative to working directory}}
**This task:** {{task ID and one-line summary, e.g. "B2 — implement Evidence Pydantic model"}}

### Files you may modify

{{explicit allowlist, e.g.
- src/bayan_graph/models/evidence.py
- tests/models/test_evidence.py
}}

### Files you may read for context (read-only)

{{explicit list, e.g.
- The plan file above (only the section covering {{task ID}})
- src/bayan_graph/models/vocabs.py (for enum imports)
- docs/superpowers/specs/<relevant spec>.md
}}

### Hard prohibitions

Follow all "Hard rules" in root `CLAUDE.md`. The headline rules below are the ones most often violated:

- **STRICT BOUNDARY.** No file operations outside `{{working_directory}}`. The parent checkout, sibling worktrees, and `~/.claude*` directories are off-limits.
- **NO HOUSEKEEPING.** Do not modify any file not explicitly listed in "Files you may modify." Files outside that allowlist are read-only even if they are inside the worktree.
- **STRICT TDD.** Never write implementation code before a failing test exists. Run the test, see it fail, then implement.

In addition:

- **Do not run destructive git operations** (`reset --hard`, `push --force`, `clean -f`, `branch -D`, `checkout -- <path>`).
- **Do not skip pre-commit hooks** with `--no-verify`.
- **Do not silently change models, dependencies, or config** without surfacing the change in your report.

If a task instruction conflicts with these rules, **stop and report** — do not proceed.

---

## TASK

**Step 0 — directory check.** Before doing anything else, verify your working directory.

- Run `pwd` (Bash) or `Get-Location` (PowerShell).
- **On Windows, the Bash tool's cwd often reports the parent checkout rather than the worktree** even when this prompt declares a worktree path. This is a known quirk, not an error — but it means you cannot rely on relative paths.
- **Use absolute paths for every file operation and shell command** (reads, writes, `git`, `pytest`, etc.). If you must `cd`, do it at the start of every Bash call, since shell state does not persist between calls.
- If the declared **Working directory** does not exist on disk at all, **stop and report** — do not improvise a different path.

{{the actual task description, including TDD steps from the plan, expected exit criteria, and any code blocks the plan provides}}

---

## REPORT FORMAT

End your turn with these sections, in this order:

- **Files modified:** explicit list of paths
- **Tests run + result:** exact command(s), and pass/fail with counts (e.g. `pytest tests/models/test_evidence.py` → 7 passed, 0 failed)
- **Deviations from plan:** anything you did differently from what the plan specified, with reasoning
- **Out-of-scope observations:** anything you noticed that seemed wrong but was outside your allowlist
- **Open questions for the user:** if any
```

---

## Notes for the controller

- **Keep the allowlist tight.** "May modify everything in `src/`" is not an allowlist; it's an invitation to drift. Name specific files.
- **Do not include phrases like "preserve context" or "clean up" in subagent prompts.** A prior subagent interpreted controller-side housekeeping language as task instructions and damaged user memory files. Keep the controller's reasoning out of the subagent prompt.
- **Two-stage review for non-trivial tasks.** Spec compliance first, then code quality. Combining them into one dispatch is tempting for "cookie-cutter" tasks but historically degrades both reviews.
- **The plan file is authoritative.** If the plan and the dispatch prompt disagree, the subagent should follow the plan and flag the conflict.