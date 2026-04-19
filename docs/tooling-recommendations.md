# Bayan-Graph — Tooling & Skills Recommendations

_Generated: 2026-04-18. Initial tooling survey before brainstorming._

## Project framing

Production RAG system for Islamic knowledge with enforced citations and structured representation of scholarly disagreement across madhahib (Hanafi, Maliki, Shafi'i, Hanbali). Stack: Claude API, LangGraph, Pinecone, RAG. Dual goal: AI Engineering portfolio signal (GCC job market + Ergon-style deployment work) and benefit to the Muslim community.

---

## Already available — use immediately

### Process (mandatory before touching design)
- `superpowers:brainstorming` — required before any creative/scoping work
- `superpowers:writing-plans` — for the multi-phase architecture
- `superpowers:writing-skills` — custom domain skills will be needed (see gaps)
- `superpowers:test-driven-development`
- `superpowers:systematic-debugging`
- `superpowers:dispatching-parallel-agents` — parallel evaluation runs

### Stack-relevant
- `claude-api` — core SDK, prompt caching (critical for RAG cost), model choice
- `api-design-principles` — query endpoint design
- `supabase-postgres-best-practices` — structured side of the data model (madhab rulings, evidence chains)

---

## To install via `npx skills add`

### LangGraph / RAG (high-install, official sources)
- `langchain-ai/langchain-skills@langgraph-fundamentals` (5.2K installs)
- `langchain-ai/langchain-skills@langgraph-persistence` (4.7K) — state across agent turns
- `langchain-ai/langchain-skills@langchain-rag` (4.7K)
- `wshobson/agents@rag-implementation` (6.5K)
- `jeffallan/claude-skills@rag-architect` (1.5K) — arch-level, not impl

### Vector DB
- `pinecone-io/skills@pinecone-docs` (official)

### Evaluation (non-negotiable for a hallucination-constrained system)
- `wshobson/agents@llm-evaluation` (5.3K)
- `supercent-io/skills-template@agent-evaluation` (10.1K)

### Knowledge graph (for madhab structure)
- `aradotso/trending-skills@understand-anything-knowledge-graph` (994)

---

## Architecture-shaping web findings

### 1. Anthropic Citations API — use as the baseline, don't rebuild
Native document chunking + citation linking; internal evals show +15% recall vs. custom implementations. Handles the hard parts of extracting relevant citations. You still implement document selection and prompt assembly — Citations handles attribution.
- [Introducing Citations on the Anthropic API](https://www.anthropic.com/news/introducing-citations-api)
- [Citations — Claude API Docs](https://docs.anthropic.com/en/docs/build-with-claude/citations)
- [Simon Willison's analysis](https://simonwillison.net/2025/Jan/24/anthropics-new-citations-api/)

### 2. Agentic RAG, not classic vector RAG
This problem (multi-madhab reasoning + citation verification + iterative evidence gathering) is the exact case where agentic wins despite higher cost. Use vector RAG only as a tool inside a LangGraph control loop, not as the top-level pipeline. Production systems route simple lookups through vector RAG for speed, complex scholarly questions through agentic search.
- [Traditional RAG vs. Agentic RAG — NVIDIA](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/)
- [Agentic RAG vs Classic RAG — Towards Data Science](https://towardsdatascience.com/agentic-rag-vs-classic-rag-from-a-pipeline-to-a-control-loop/)

### 3. Prior art to read BEFORE brainstorming
Others have been in this problem space. Read these first to sharpen scope and avoid re-deriving known patterns.
- [**FARSIQA: Faithful RAG for Islamic QA**](https://arxiv.org/abs/2510.25621) — FAIR-RAG architecture: adaptive query decomposition, evidence sufficiency assessment, iterative sub-query loop. Closest match to the goal.
- [**Graph-Based RAG with Quranic Ontology**](https://medium.com/@maulanaanab/trustworthy-islamic-agent-a-graph-based-rag-for-islamic-content-using-a-quran-ontology-be7e1d729315) — parallel semantic + graph retrieval; validates the structured-knowledge approach.
- [**Transformer Tafsir (QIAS 2025)**](https://arxiv.org/html/2509.23793) — hybrid retrieval on Islamic knowledge QA shared task.
- [**IslamicMMLU benchmark**](https://arxiv.org/html/2603.23750) — use as eval set; don't build one from scratch.
- [**AI Fiqh platform**](https://dev.to/irfanghapar/ai-fiqh-retrieval-augmented-generation-rag-nl8) — relevant to the scholar-verification layer if it gets built.
- **EMasjid** incorporates Al-Hidayah (Hanafi), Al-Mudawwanah Al-Kubra (Maliki), Al-Umm (Shafi'i), Al-Mughni (Hanbali) as primary texts per madhab.

---

## Gaps — skills to write yourself

No existing skill covers:
- Arabic Quran/Hadith text normalization (diacritics, variant spellings, transliteration)
- Madhab citation hierarchy enforcement: Quran > authentic Hadith > ijma' > qiyas > madhab opinion
- Isnad (chain-of-transmission) representation for hadith authenticity grading
- Multi-madhab disagreement structure (typed records: per-madhab ruling + evidence chain + dissent)

Use `superpowers:writing-skills` to produce these. This is where domain literacy becomes durable IP — non-Arabic-speaking candidates cannot fake these.

---

## Recommended sequence

1. Install LangGraph + RAG + evaluation skills (done in this session)
2. Read FARSIQA paper
3. Skim graph-based RAG Medium article
4. Invoke `superpowers:brainstorming` — with prior art loaded, the scoping conversation will be sharper
5. Write plan via `superpowers:writing-plans`
