# Bayan-Graph — Agent Guardrails

This file is loaded automatically into every Claude Code session opened in this repo (and into every worktree). Read it before taking any action.

## Project at a glance

Bayan-Graph is a production RAG system for Islamic fiqh with two hard requirements:

1. Every answer cites Quran / Hadith / madhab texts via the **Anthropic Citations API** — no synthesis without source anchors.
2. Scholarly disagreement across the four Sunni madhahib (Hanafi, Maliki, Shafiʿi, Hanbali) is modeled as **structured data**, not free-form prose.

**Stack:** Claude API (`claude-sonnet-4-6` synthesis, `claude-haiku-4-5-20251001` classifier) · LangGraph · Pinecone · Cohere embed/rerank · Python 3.12 · uv.

## Workspace layout

- `main` branch — plans, specs, top-level docs, and this file. **Do not commit implementation code to `main`.**
- Implementation work happens in **git worktrees** under `.worktrees/<branch>/` on feature branches like `feat/plan-01-foundation`.
- The active plan lives at `docs/superpowers/plans/2026-04-22-bayan-graph-plan-01-foundation-walking-skeleton.md`. Every task references it.
- Specs live under `docs/superpowers/specs/`.
- Subagent dispatch template: `docs/subagent-dispatch-template.md`. Every `Agent` call in this project must use it.

## Hard rules — non-negotiable

### Scope boundaries

- **Positive pathing.** Every task comes with an explicit allowlist of files you may modify. Files not in that allowlist are **strictly read-only**, even if they are inside the worktree.
- **Never read, write, or delete files outside the worktree you are working in.** If you are working in `.worktrees/plan-01-foundation`, do not touch anything in the parent checkout, sibling worktrees, or any other path on disk.
- **Never touch user memory or config directories.** This includes `~/.claude/`, `~/.claude-personal/`, `~/.claude-ergon/`, and any path containing `memory/` or `auto-memory/`. Those belong to the user, not to agents.
- **Never modify repo-root documentation** (`CLAUDE.md`, `README.md`, files under `docs/`, plans, specs) unless that file is in your task's allowlist. These are project infrastructure, not implementation surface.
- If you find files outside your scope that look "wrong" or "out of place," **leave them alone and surface the observation to the user.** Do not "clean up."

### No meta-tasks

Do not perform "cleanup," "context pruning," "memory optimization," or any housekeeping activity unless that is the **primary task** in your dispatch prompt. A task to implement a Pydantic model is not a license to tidy adjacent files, refactor unrelated imports, or trim documentation. Stay on task.

### Destructive operations require explicit user approval

Before running any of the following, stop and ask:

- `rm` / `Remove-Item` on any file you did not create in this session
- `git reset --hard`, `git push --force`, `git checkout -- <path>`, `git clean -f`, `git branch -D`
- Overwriting a file you have not read in this session
- Modifying CI config, pre-commit hooks, or `.gitignore` in a way that hides existing tracked files

The cost of asking is one round-trip. The cost of an unwanted destructive action is lost work.

### TDD discipline

Per the active plan, every node and model is implemented red → green:

1. Write the failing test first.
2. Run it. Confirm it fails for the right reason.
3. Implement the minimum code to pass.
4. Run again. Confirm green.
5. Commit.

Do not write implementation before the test exists.

### Citations are load-bearing, not decorative

Synthesis nodes that produce user-facing answers must use the **Citations API**. A passing test that does not exercise citation anchoring is not a passing test for an answer-producing node.

## Subagent dispatch — controller responsibilities

Every `Agent` tool call in this project must prepend the standard scope header from `docs/subagent-dispatch-template.md`. **The controller (main thread) is responsible for this.** If you are about to dispatch a subagent and have not pasted in the scope header, stop and add it.

A subagent that goes off-task and edits files outside its allowlist is a controller failure, not just a subagent failure — the prompt was missing guardrails.

## Model strategy

- **Implementer / reviewer subagents:** Sonnet (`claude-sonnet-4-6`) by default.
- **Opus fallback:** only for H1 / H2 LangGraph integration tasks if Sonnet stalls.
- Do not silently upgrade to Opus to "be safe" — that costs the user money. Ask first.

## When in doubt

Ask. The user prefers a clarifying question to a confidently wrong action.