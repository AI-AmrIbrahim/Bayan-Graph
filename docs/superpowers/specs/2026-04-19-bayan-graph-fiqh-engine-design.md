# Bayan-Graph Phase 1 — Multi-Madhab Fiqh Engine

**Status:** Design approved, pending spec review before planning.
**Date:** 2026-04-19.
**Scope:** Phase 1 only. Phase 2 (general Islamic knowledge base — tafsir, hadith narrative QA, aqeedah) is a separate spec written after Phase 1 ships.

---

## 1. Executive summary

A retrieval-augmented generation system that answers ibadat (worship) fiqh questions with a structured, per-madhab comparative ruling across the four Sunni madhahib (Hanafi, Maliki, Shafi'i, Hanbali). Every ruling is backed by a ranked evidence chain (Quran → mutawatir/sahih hadith → ijma → qiyas → madhab articulation), and every citation is mechanically validated against the source corpus before reaching the user.

The system is built on Claude API (Citations API for citation anchoring), LangGraph (for the agentic control loop), and Pinecone (for vector retrieval). The output is structured JSON (the source of truth) with a prose narrative layer rendered on top. A 75-question hand-curated evaluation set, anchored by Ibn Rushd's *Bidayat al-Mujtahid* and each madhab's primary text, provides deterministic scoring.

Alongside the product, three standalone Claude Code skills are extracted and published: Arabic text normalization, madhab citation hierarchy enforcement, and hadith chain/matn handling.

---

## 2. Scope and locked constraints

These constraints were settled during brainstorming and are load-bearing for the design that follows.

| Constraint | Decision |
|---|---|
| Primary use case | Multi-madhab comparative fiqh engine (option A). Phase 2 = general knowledge base. |
| Fiqh scope | Ibadat only (purification, prayer, fasting, zakat, hajj). |
| Madhab coverage | All four Sunni madhahib from day 1. |
| Corpus | Quran + Sahih Bukhari + Sahih Muslim + 4 Sunan (Abu Dawud, Tirmidhi, Nasa'i, Ibn Majah) + Al-Hidayah (Hanafi) + Al-Mudawwanah (Maliki) + Al-Umm (Shafi'i) + Al-Mughni (Hanbali) + Ibn Rushd's *Bidayat al-Mujtahid* (for comparative/ikhtilaf reasoning). Ibadat chapters only. |
| Output | Structured JSON as source of truth; prose narrative rendered on top. |
| Query language | English queries only (MVP); bilingual display (Arabic source + English translation always shown for citations); multilingual embeddings at ingestion. |
| Evaluation | Automated citation faithfulness + ~75-question hand-curated ibadat eval set. Ibn Rushd as ground-truth anchor, cross-referenced with each madhab's own book. |
| Deployment | FastAPI backend + Next.js web UI (public demo). Arabic-normalization, madhab-citation-hierarchy, hadith-chain-handling skills published separately. |
| Scope control | Upfront query classifier with categorized refusal for out-of-scope queries. `refusal_reason` is a first-class answer shape. |

---

## 3. Architecture overview

### Services and infrastructure

```
┌────────────────┐          ┌────────────────┐
│  Next.js UI    │ ──HTTP──►│  FastAPI API   │
│  (Vercel)      │          │  (Fly.io or    │
│                │          │   Render)      │
└────────────────┘          └───────┬────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                ┌────────────────┐     ┌────────────────┐
                │  LangGraph     │     │  Claude API    │
                │  runtime       │◄───►│  (Citations,   │
                │  (in-process)  │     │   Haiku/Sonnet)│
                └────────┬───────┘     └────────────────┘
                         ▼
                ┌────────────────┐     ┌────────────────┐
                │  Pinecone      │     │  Postgres      │
                │  (7 namespaces)│     │  (Supabase)    │
                │                │     │  checkpointer  │
                │                │     │  traces, eval  │
                └────────────────┘     │  matn_clusters │
                                       └────────────────┘
```

**Pinecone namespaces (one index):** `quran`, `hadith`, `hanafi`, `maliki`, `shafii`, `hanbali`, `comparative` (Ibn Rushd).

**Rationale for separate namespaces:** madhab-scoped retrieval becomes a routing decision rather than a metadata scan, and per-madhab parallelism in the control loop is trivial.

**Postgres holds:** LangGraph checkpointer state, retrieval trace rows, eval run history, matn cluster metadata (see §5), `needs_review` queue for low-confidence ingestion chunks.

### Repository layout

```
bayan-graph/
├── api/          # FastAPI, LangGraph nodes, retrieval, synthesis
├── ingestion/    # One-shot scripts: chunker, embedder, role tagger, indexer
├── eval/         # Eval set YAML, harness, automated faithfulness checks
├── ui/           # Next.js, read-only client of the API
├── skills/       # Published custom skills (extracted at MVP launch)
└── docs/         # Specs, runbooks, ADRs
```

Monorepo keeps ingestion, API, and eval sharing schema/citation code. UI deploys independently but lives in the same repo for atomic change tracking.

### Out of scope for Phase 1

Auth, rate limiting beyond per-IP cap, query history, user accounts, streaming responses, conversation memory / multi-turn, query rewriting before classification, Arabic query input, Neo4j / graph-database layer, narrator-level isnad graphing, IslamicMMLU integration as a primary metric. All of these are Phase 1.5+.

---

## 4. Data model and ingestion pipeline

### Source acquisition

| Source | Format | Provenance |
|---|---|---|
| Quran (Arabic Uthmani) | Per-ayah JSON | tanzil.net |
| Quran (English translations) | Saheeh International + Pickthall | tanzil.net, aligned by surah:ayah |
| Hadith (6 collections) | Per-hadith JSON with isnad/matn/grade | sunnah.com JSON dumps, mirrored locally |
| Al-Hidayah (Hanafi) | OCR'd Arabic + partial English | shamela.ws |
| Al-Mudawwanah (Maliki) | Arabic | shamela.ws |
| Al-Umm (Shafi'i) | Arabic + partial English | shamela.ws |
| Al-Mughni (Hanbali) | Arabic | shamela.ws |
| *Bidayat al-Mujtahid* (Ibn Rushd) | Arabic + English (Nyazee translation) | Published digital editions |

Raw sources cached to `ingestion/raw/` with provenance metadata (source URL, retrieved date, checksum). Ingestion is idempotent; re-runs do not duplicate.

### Chunking strategies

Three distinct strategies, because natural units differ across source types.

**Quran — one chunk per ayah, windowed context.**
Chunk text for embedding = Arabic ayah (normalized) + English translation + preceding ayah + following ayah. Display surfaces only the ayah itself; the window is for retrieval signal.
Metadata: `{surah, ayah, juz, revelation_place, translations, topic_tags, abrogation}`.

**Hadith — one chunk per occurrence (collection + number), matn-only embedding.**
Isnad stored as structured metadata, not embedded. Each occurrence is its own chunk; same matn across collections is a separate chunk per occurrence, linked via `matn_cluster_id`.
Per-occurrence metadata: `{collection, number, book, matn_cluster_id, chain_summary, chain_grade, text_ar, text_en, abrogation, topic_tags}`.

**Madhab books and *Bidayat al-Mujtahid* — semantic section chunks with hierarchical breadcrumbs.**
Chunk size 600-1000 tokens, 100-token overlap, `RecursiveCharacterTextSplitter`-driven.
Metadata: `{madhab, book, volume, chapter, section, toc_path, represents_stance_of, role_in_source, cites, normalization}`.

### Arabic normalization (ingestion-side)

Applied only to the embedding-input copy; original Arabic preserved verbatim for display and citation. Steps: strip harakat, normalize letter variants (`أ إ آ → ا`, `ى → ي`, `ة → ه` context-sensitive, `ؤ → و`, `ئ → ي`), remove tatweel, normalize whitespace and punctuation.

**Quality score + flags** per chunk: `normalization: {score: 0.0-1.0, flags: [ambiguous_hamza, diacritic_stripped_ambiguous, ocr_confidence_low, ...]}`. Chunks scoring below threshold (~0.85) are routed to a `needs_review.jsonl` queue for human triage without blocking ingestion. Corpus health metric on eval dashboard: percentage of corpus at high confidence vs needs-review.

This logic is packaged as the first published skill (`bayan-arabic-normalization`, §9).

### Table-of-contents as faceted tags

The author's own chapter structure is authoritative domain knowledge. Every madhab-book and *Bidayat* chunk carries `toc_path: ["Kitab al-Taharah", "Bab al-Wudu", "Fasl al-Furud"]` — array, filterable, sortable at retrieval time. For books with damaged TOC (OCR issues, older prints), `toc_source: author | inferred | none` so the retriever knows how much to trust the tag.

### Role tagging on madhab-book chunks (ingestion-time)

Each madhab-book chunk is tagged with two fields beyond `madhab`:

- `represents_stance_of: [madhab, ...]` — whose position the chunk's content actually represents (not necessarily the host book's madhab — Al-Hidayah frequently quotes Al-Shafi'i to refute).
- `role_in_source` — controlled vocab: `madhab_position`, `supporting_daleel`, `opposing_view_cited`, `opposing_view_cited_and_refuted`, `comparative_discussion`, `preparatory_context`.
- `refuted_in_host_book: bool`.

**Extraction approach:** heuristic pattern parser on classical Arabic rhetorical markers (`قال الشافعي...ولكن`, `وبهذا نقول`, `ذهب الجمهور إلى...`, etc.) + LLM backstop for ambiguous cases. Ambiguous chunks join `needs_review`. `toc_path` helps too — chapters titled *Bab Ma Ikhtalafa Fihi al-A'immah* are comparative by definition.

### Cross-reference extraction (madhab-chunk `cites` field)

Madhab books themselves cite their evidence inline ("the basis for this ruling is Allah's saying [Quran 5:6] and the hadith narrated by Uthman"). During madhab-book ingestion, these inline citations are parsed and stored on the chunk:

```json
"cites": [
  {"type": "quran", "ref": "5:6"},
  {"type": "hadith", "collection": "bukhari", "number": 164}
]
```

This is automated (pattern extraction from source text), bidirectional naturally, and used by the hierarchy sort (§6) and synthesis (§6) nodes to boost and cross-validate evidence retrieved separately.

### Hadith matn clusters (separate Postgres table)

A single matn often appears across multiple collections via different chains. Cluster metadata is stored once in Postgres `matn_clusters` and referenced by `matn_cluster_id` from each occurrence chunk:

```json
{
  "cluster_id": "cluster:masah_khuff:jarir",
  "canonical_text_ar": "...",
  "canonical_text_en": "...",
  "tawatur_status": "mashhur",
  "chain_count_total": 14,
  "chain_count_independent": 4,
  "best_occurrence": "hadith:bukhari:203",
  "occurrences": ["hadith:bukhari:203", "hadith:muslim:274", "hadith:abudawud:149"],
  "scholarly_consensus": "sahih"
}
```

**`tawatur_status` controlled vocab:** `mutawatir_lafzi`, `mutawatir_ma'nawi`, `mashhur`, `aziz`, `gharib`, `ahad` (default).

Tawatur flags imported from reference lists (al-Suyuti's *Qatf al-Azhar al-Mutanathira*, al-Kattani's *Nazm al-Mutanathir*) as a one-time ingestion stage.

### Hadith chain grading (per-occurrence)

Each occurrence carries:

```json
"chain_grade": {
  "grade": "sahih",
  "source": "collection_internal",
  "disputes": [
    {"source": "shuayb_arnaout", "grade": "hasan"},
    {"source": "ahmad_shakir", "grade": "sahih_li_ghayrihi"}
  ]
}
```

`source` controlled vocab: `collection_internal`, `shuayb_arnaout`, `albani`, `ahmad_shakir`, `ibn_hajar`, `manual_reference_work`.

When graders disagree, the effective grade used downstream is conservative: disputed sahih/da'if → treated as `hasan_pending`, retained but flagged.

### Abrogation handling (ingestion-time)

Each hadith and ayah chunk carries:

```json
"abrogation": {
  "status": "none | abrogated | abrogating | disputed_abrogation",
  "related_ref": "...",
  "scholarship_source": "..."
}
```

Coverage is incomplete for MVP (this is a sub-field of its own); even partial tagging catches common cases. The eval harness includes a specific regression test: abrogated evidence cannot appear as `role: primary_evidence`.

### Explicitly scoped OUT for MVP (hadith handling)

- Automated chain-independence analysis (`chain_count_independent` is imported, not derived).
- Narrator biography database (no Tahdhib al-Kamal, Mizan al-I'tidal, etc.).
- Chain-level tamarud (conflict) resolution.
- Hadith not in Kutub al-Sittah (if cited by a madhab book, captured as a madhab-chunk citation, not a retrievable hadith chunk).

These form the cleanest Phase 1.5 extension path and are where a dedicated graph database becomes earned.

### Embedding model and index

- **Model:** Cohere `embed-multilingual-v3.0`, 1024 dims. Input type `search_document` at ingestion, `search_query` at runtime.
- **Pinecone:** serverless tier, cosine metric, one index with 7 namespaces. Estimated corpus size for ibadat scope: ~25k–30k vectors.
- **Batching:** 96 chunks per embedding request.
- Embedding call abstracted behind `embed()` for future swap (candidates: voyage-multilingual-2, bge-m3).

### Ingestion pipeline stages

```
acquire_raw → parse_to_records → normalize_arabic → align_translations
  → import_gradings → cluster_matns → import_tawatur_flags
  → tag_roles (madhab books) → extract_cites (madhab books)
  → chunk (per-strategy) → embed (batched) → upsert_to_pinecone
```

Each stage is a pure function with cached intermediate outputs in `ingestion/cache/`. Re-running after a strategy change re-chunks/re-embeds only downstream of the changed stage. CLI: `python -m ingestion run --source quran --stage all`.

---

## 5. Answer JSON schema (the contract)

Every other component produces, consumes, or validates against this schema. It is the source of truth; the narrative prose is rendered from it.

```json
{
  "query_id": "uuid",
  "query": "Is it permissible to wipe over socks during wudu?",
  "classification": {
    "scope": "in_scope_ibadat",
    "topic_path": ["taharah", "wudu", "masah_on_footwear"],
    "confidence": 0.94
  },
  "answer_type": "multi_madhab_ruling",

  "positions": [
    {
      "madhab": "hanafi",
      "primary_view": {
        "authority_level": "mu'tamad",
        "stance": "permissible",
        "stance_nuance": "permissible with specific conditions (khuff only, time-limited)",
        "ruling_text": {"ar": "...", "en": "..."},
        "attributed_to": ["Abu Hanifa", "Abu Yusuf", "Muhammad al-Shaybani"],
        "era": "foundational",
        "evidence": [
          {
            "rank": 1,
            "type": "quran",
            "ref": {"surah": 5, "ayah": 6},
            "text": {"ar": "...", "en": "..."},
            "role": "contextual",
            "chunk_id": "quran:5:6"
          },
          {
            "rank": 2,
            "type": "hadith",
            "ref": {
              "collection": "bukhari",
              "number": 203,
              "book": "wudu",
              "matn_cluster_id": "cluster:masah_khuff:jarir",
              "tawatur_status": "mashhur",
              "sibling_occurrences": [
                {"collection": "muslim", "number": 274},
                {"collection": "abudawud", "number": 149}
              ]
            },
            "grade": {
              "effective": "sahih",
              "per_chain": {"bukhari_chain": "sahih"},
              "disputed_by": []
            },
            "narrator_chain_summary": "Jarir ibn Abdullah",
            "text": {"ar": "...", "en": "..."},
            "role": "primary_evidence",
            "chunk_id": "hadith:bukhari:203"
          },
          {
            "rank": 3,
            "type": "madhab_text",
            "ref": {
              "book": "al-hidayah",
              "toc_path": ["Kitab al-Taharah", "Bab al-Masah ala al-Khuffayn"],
              "volume": 1, "page": 31
            },
            "text": {"ar": "...", "en": null},
            "role": "madhab_articulation",
            "chunk_id": "hanafi:hidayah:v1:31",
            "cites": [
              {"type": "quran", "ref": "5:6"},
              {"type": "hadith", "collection": "bukhari", "number": 203}
            ]
          }
        ],
        "reasoning": "The Hanafi madhab permits wiping over khuff based on the mutawatir hadith..."
      },
      "alternate_views": []
    },
    { "madhab": "maliki",  "...": "..." },
    { "madhab": "shafii",  "...": "..." },
    { "madhab": "hanbali", "...": "..." }
  ],

  "cross_madhab_analysis": {
    "points_of_agreement": [
      "All four madhahib permit wiping over leather socks (khuff) in principle.",
      "All four require prior wudu before donning the footwear."
    ],
    "points_of_disagreement": [
      {
        "dimension": "duration_for_resident",
        "positions": {
          "hanafi": "one day and night",
          "maliki": "no fixed limit (some narrations)",
          "shafii": "one day and night",
          "hanbali": "one day and night"
        },
        "reasons_for_ikhtilaf": [
          {
            "type": "hadith_authenticity",
            "description": "Malik did not adopt the time-limiting hadith of Safwan ibn 'Assal with the same weight...",
            "affected_madhahib": ["maliki"],
            "supporting_evidence": [{"type": "madhab_text", "chunk_id": "maliki:mudawwanah:..."}]
          },
          {
            "type": "usul_principle",
            "description": "Hanafis apply istihsan to restrict duration...",
            "affected_madhahib": ["hanafi"],
            "supporting_evidence": [{"type": "madhab_text", "chunk_id": "hanafi:hidayah:..."}]
          }
        ]
      }
    ],
    "consensus_level": "majority_with_dissent"
  },

  "narrative": "English prose rendering of the above.",

  "citation_audit": {
    "total_citations": 18,
    "resolved": 18,
    "text_match_verified": 18,
    "hierarchy_violations": 0,
    "unverified": []
  },

  "refusal_reason": null,

  "meta": {
    "model_versions": {"classifier": "haiku-4-5", "synthesis": "sonnet-4-6"},
    "retrieval_iterations": 1,
    "latency_ms": 11840,
    "token_usage": {"input": 28500, "output": 3200, "cached_read": 24000}
  }
}
```

### Schema invariants (validator-enforced)

1. **`refusal_reason` and `positions` are mutually exclusive.** Out-of-scope queries populate `refusal_reason`, empty `positions`. In-scope queries populate `positions`, `refusal_reason = null`.
2. **`evidence[].rank` is assigned by Node E (hierarchy sort), not the LLM.** LLM reads rank; does not choose it.
3. **`citation_audit` is populated by Node V (post-synthesis validator),** not the LLM.
4. **`authority_level` controlled vocab:** `mu'tamad`, `mashhur`, `marjuh`, `shadh`, `disputed_within_madhab`.
5. **`role` controlled vocab:** `primary_evidence`, `contextual`, `madhab_articulation`, `corroborating`, `contested`.
6. **Ikhtilaf reason types (controlled vocab, published via skill 2):** `hadith_authenticity`, `hadith_preference_when_conflicting`, `textual_interpretation`, `usul_principle`, `qiyas_application`, `scope_of_text`, `linguistic`, `preference_of_athar`, `historical_evolution`.
7. **Within a madhab position:** `primary_view` is required if the madhab has any evidence; `alternate_views[]` may be empty or contain any number. When no clear primary-vs-alternate signal exists, `authority_level: disputed_within_madhab` and views are surfaced flat.

---

## 6. LangGraph nodes and control flow

### State schema

One `TypedDict` flows through every node. Fields are populated progressively; later nodes read earlier fields, never mutate them.

```python
class BayanState(TypedDict):
    # Input
    query_id: str
    query: str

    # Stage A
    classification: Classification | None

    # Stage B / D
    retrieved: dict[str, list[Chunk]]  # keyed by namespace
    retrieval_iterations: int          # hard-capped at 2

    # Stage C
    sufficiency: dict[Madhab, SufficiencyResult] | None

    # Stage E
    ranked_evidence: dict[Madhab, list[EvidenceItem]] | None

    # Stage F / G / H
    positions: list[Position] | None
    cross_madhab_analysis: CrossMadhabAnalysis | None
    narrative: str | None

    # Validation
    citation_audit: CitationAudit | None

    # Terminal
    refusal_reason: str | None         # mutually exclusive with positions

    # Meta
    meta: RunMeta
```

### Nodes

| Node | Reads | Writes | LLM? | Parallel? | Notes |
|---|---|---|---|---|---|
| **A** `classify_query` | `query` | `classification`, `refusal_reason?` | Haiku, cached | no | Scope gate. Terminates on out-of-scope. |
| **B** `parallel_retrieve` | `query`, `classification.topic_path` | `retrieved` | no (pure retrieval + reranker) | yes — `Send` × 6 (quran, hadith, 4× madhab). Comparative namespace and the refuted-chunks secondary pass are driven by Node G, not B. | Vector top-k → rerank → top-k-reranked. See §7. |
| **C** `check_sufficiency` | `retrieved`, `classification` | `sufficiency` | Haiku, cached | yes — `Send` × 4 | Returns `{sufficient, missing, suggested_reformulation, irrelevant_chunk_ids, out_of_context_chunk_ids}` per madhab. |
| **D** `targeted_reretrieve` | `sufficiency`, `retrieval_iterations` | `retrieved` (append), `retrieval_iterations += 1` | no | yes — only insufficient madhahib | Hard cap: 2 iterations. On cap, proceed with `sufficiency.capped: true`. |
| **E** `hierarchy_sort` | `retrieved`, `classification` | `ranked_evidence` | no (deterministic) | no | Quran > mutawatir > sahih mashhur > sahih ahad > hasan > ijma > qiyas > da'if > madhab_text. Drops irrelevant chunks from C. Filters madhab namespaces by `represents_stance_of` and `role_in_source`. |
| **F** `synthesize_positions` | `ranked_evidence` | `positions` | Sonnet + Citations API, cached system | yes — `Send` × 4 | Each madhab fans out. Sub-step: extract authority_level from signal phrases. Produces `primary_view` + `alternate_views[]`. |
| **G** `cross_madhab_analysis` | `positions`, `retrieved.comparative`, secondary-pass refuted chunks from madhab namespaces | `cross_madhab_analysis` | Sonnet, cached | no | Uses Ibn Rushd `comparative` chunks + a secondary retrieval across madhab namespaces filtered to `role_in_source ∈ {opposing_view_cited, opposing_view_cited_and_refuted, comparative_discussion}`. Produces agreement/disagreement with `reasons_for_ikhtilaf[]`. |
| **H** `narrative_render` | full state | `narrative` | Sonnet, cached | no | Prose rendering. Introduces no new citations; only references IDs from `positions`. |
| **V** `validate_citations` | `positions`, `narrative` | `citation_audit` | no (deterministic) | no | Five-check audit (§7). Triggers three-tier escalation on failure. |

### Edges

```
START
  └─► A (classify_query)
        │
        ├─[scope != in_scope]─► TERMINAL (refusal shape)
        │
        └─[in_scope]─► B (parallel_retrieve via Send × 7)
                         │
                         ▼
                       C (sufficiency via Send × 4)
                         │
                         ├─[any insufficient && iters < 2]─► D ──► C (loop)
                         │
                         └─[all sufficient || cap hit]─► E (hierarchy_sort)
                                                          │
                                                          ▼
                                                        F (synthesize × 4)
                                                          │
                                                          ▼
                                                        G (cross_madhab_analysis)
                                                          │
                                                          ▼
                                                        H (narrative_render)
                                                          │
                                                          ▼
                                                        V (validate_citations)
                                                          │
                                                          ├─[unverified > 0]─► escalation policy (§7)
                                                          │
                                                          └─► TERMINAL
```

### LangGraph specifics

- **Parallelism via `Send`.** Stages B, C, F fan out across namespaces/madhahib. Reducer on `retrieved` and `sufficiency` is merge-by-key.
- **Checkpointer on Postgres.** Every state transition checkpointed. Powers eval deterministic replay, debugging, and `langgraph-persistence` skill usage. Keyed by `query_id` + `retry_count`.

---

## 7. Retrieval, reranking, and citation enforcement

### Retrieval config per namespace

| Namespace | Vector top-k | Rerank-k | Metadata filters (when classification confident) |
|---|---|---|---|
| `quran` | 10 | 4 | `surah`, `juz`, `topic_tag` |
| `hadith` | 15 | 5 | `collection`, `grade∈{sahih,hasan}` (prefer), `topic_tag` |
| `hanafi` | 12 | 5 | `toc_path` prefix, `represents_stance_of:[hanafi]`, `role_in_source∈{madhab_position, supporting_daleel}` |
| `maliki` | 12 | 5 | analogous |
| `shafii` | 12 | 5 | analogous |
| `hanbali` | 12 | 5 | analogous |
| `comparative` | 10 | 4 | `madhahib_addressed ⊇ {hanafi, maliki, shafii, hanbali}` |

**Reranker:** Cohere `rerank-multilingual-v3.0` as a sub-step inside Node B. Flow: `embed → vector_topk (k=20) → rerank(query, chunks) → topk_reranked (k=5)`. The main noise reducer for topical adjacency failures.

**Filter rationale (madhab namespaces):** Default filter drops `opposing_view_cited` and `opposing_view_cited_and_refuted` chunks from a madhab's own evidence pile during Node B retrieval for Node F synthesis. Those chunks are explicitly accessed by Node G through a secondary retrieval pass that queries all four madhab namespaces with the inverted filter `role_in_source ∈ {opposing_view_cited, opposing_view_cited_and_refuted, comparative_discussion}`. The `comparative` namespace (Ibn Rushd's *Bidayat al-Mujtahid*) is the primary source for Node G's ikhtilaf reasoning; the refuted-chunks secondary pass supplements it with each madhab's own articulation of what they reject and why — exactly the signal for populating `reasons_for_ikhtilaf[]`.

**No HyDE or multi-query expansion for MVP.** No sparse/dense hybrid for MVP (Arabic tokenization complexity; reranker covers most exact-phrase misses). Revisit only if eval surfaces specific recall failures.

### Query preparation

Raw query embedded with `search_query` input-type. Topic path from classification passed as a soft metadata filter (boosts; does not exclude). Query reformulations from Node D are re-embedded and results merged with the original.

### Citation enforcement via Claude Citations API

Node F uses the Citations API rather than prompt-based citation instructions. Flow:

```
documents = [chunk as citable doc for chunk in ranked_evidence[madhab]]
response = claude.messages.create(
  model="sonnet-4-6",
  system=SYNTHESIS_SYSTEM_PROMPT,   # cached
  messages=[{role: user, content: query + schema_instructions + documents}],
  citations_enabled=True
)
```

The model does not generate citation refs as free text; the Citations API attaches refs to generated spans, anchored to input documents. Mechanically prevents fabricated-citation failures. Misquoting and mis-roled citations are caught by Node V.

Aggressive prompt caching: `SYNTHESIS_SYSTEM_PROMPT` is identical across queries and across 4 madhab fan-outs — target cache hit rate > 90%.

### Post-synthesis citation audit (Node V)

Five deterministic checks per citation. Any failure populates `citation_audit.unverified[]`:

| # | Check | Failure mode caught |
|---|---|---|
| 1 | `chunk_id` exists in `state.ranked_evidence` | Fabricated reference |
| 2 | `text.ar` matches source chunk (normalized, similarity ≥ 0.98) | Quote drift / silent paraphrase |
| 3 | `type` matches chunk's `source_type` | Mis-typed citation |
| 4 | `ref` fields match chunk metadata | Mis-ref'd citation |
| 5 | Hierarchy consistency: `role: primary_evidence` + `type: madhab_text` requires at least one Quran/sahih-hadith citation at rank ≤ this one | Madhab opinion used as primary evidence without scriptural anchor |

### Three-tier escalation on unverified citations

1. **First failure → regenerate once** with unverified IDs called out in a follow-up message.
2. **Second failure → degraded mode.** Drop unverified citations, flag the affected `Position` with `degraded: true, degradation_reason`, continue. Response reaches user with visible warning banner.
3. **Third failure (or post-regen hierarchy violation) → hard refusal.** `refusal_reason: "citation_audit_failed"`, positions empty. Rare; logged aggressively as a corpus gap signal.

### Observability: retrieval traces

Every retrieval and rerank call writes a trace row to Postgres: `{query_id, namespace, query_embedding_hash, vector_hits[], rerank_scores[], final_topk[], latency_ms}`. Powers eval replay ("what changed between runs?"), bad-answer debugging, and corpus health reports.

---

## 8. Evaluation harness

### Principles

1. Deterministic checks wherever possible. Field match, text match, citation existence, schema shape.
2. LLM-judge only for prose quality — Phase 1.5, not MVP.
3. Per-check binary, composite weighted per question.

### Eval set shape (YAML)

```yaml
id: q023
chapter: taharah
topic_path: [taharah, wudu, masah_on_footwear]
difficulty: ikhtilaf_with_dissent
query: "Is it permissible to wipe over socks during wudu?"
ground_truth_sources:
  - {work: bidayat_al_mujtahid, vol: 1, section: "Kitab al-Taharah, Bab al-Masah"}
  - {work: al_mughni,           vol: 1, section: "Masah al-Khuffayn"}

classification_expected:
  scope: in_scope_ibadat
  topic_path_must_include: [taharah, wudu]

positions_expected:
  hanafi:
    primary_view:
      stance: permissible
      stance_must_convey: [khuff_only, time_limited_24h_resident, 72h_traveler]
      authority_level: mu'tamad
      primary_evidence_types_any_of: [quran, hadith]
      primary_evidence_minimum_grade: sahih
      required_citations_any_of:
        - {type: hadith, matn_cluster_id: cluster:masah_khuff:jarir}
        - {type: hadith, matn_cluster_id: cluster:masah_khuff:safwan}
    alternate_views_allowed: true
  # ...maliki, shafii, hanbali

ikhtilaf_expected:
  dimensions_any_of: [duration_for_resident, footwear_type_extensibility]
  reasons_any_of: [hadith_authenticity, usul_principle, textual_interpretation]

refusal_expected: null

regression_flags:
  - matn_cluster_dedup
  - time_limit_vs_unlimited_disagreement_surfaced
```

**`_any_of` everywhere** instead of exact match: real rulings have acceptable variation; hard equality would produce false failures.

**`stance_must_convey` as keyword list:** the system's stance_nuance prose varies; what matters is coverage of required semantic content. Checked with bounded keyword/phrase match over prose + structured fields combined.

### Set composition — 75 questions

| Category | Count | Purpose |
|---|---|---|
| Mainstream consensus (all 4 agree) | ~25 | Baseline — no fabricated disagreement |
| Majority-with-dissent | ~25 | The happy path for the design |
| Fully contested | ~10 | 2-2 or 1-1-1-1 splits |
| Internal-madhab ikhtilaf | ~5 | Tests `alternate_views[]` |
| Refusal/scope | ~5 | Aqeedah, contemporary, non-ibadat |
| Regression tests (noise modes) | ~5 | Failure modes from §4 + tawatur + abrogation |

Distributed ~15 per chapter across taharah, salah, sawm, zakat, hajj.

### Construction process

1. Topic sampling — 15 sub-topics per chapter.
2. Question mining from Ibn Rushd + each madhab's primary book.
3. Ground-truth authoring — ~20-30 min per question × 75 ≈ 25-35 hours. The labor; the IP.
4. Schema validation — every YAML validates against Pydantic; CI gate.
5. Second-reviewer pass — `CONTRIBUTING_EVAL.md` makes it possible for domain-literate contributors to extend/review over time.

### Automated checks — five tiers

**Tier 1 — Structural (gate).** JSON schema valid; `classification.scope` matches; `positions` complete or refusal justified; no null required fields. Binary pass/fail.

**Tier 2 — Citation faithfulness (gate).** `citation_audit.unverified` empty; `hierarchy_violations == 0`; cluster refs resolve; sibling_occurrences cross-verify.

**Tier 3 — Per-madhab correctness (scored).** Per madhab: stance match (0/1), stance nuance coverage ([0,1]), authority level match (0/1), primary evidence types intersection non-empty (0/1), required citation coverage ([0,1]). Weighted sum per madhab; mean across 4.

**Tier 4 — Ikhtilaf correctness (scored).** Dimension coverage ([0,1]), reason-type coverage ([0,1]), no fabricated disagreement on consensus questions (hard failure if violated).

**Tier 5 — Regression flags (binary per flag).** Examples: `matn_cluster_dedup`, `tawatur_promotion`, `abrogation_not_cited_as_primary`, `refuted_opposing_view_not_in_madhab_pile`, `time_limit_vs_unlimited_disagreement_surfaced`.

**Composite:**
```
composite = gate(T1) * gate(T2) * (0.5 * T3 + 0.3 * T4 + 0.2 * mean(T5 flags))
```

One number per question in [0, 1]. Run-level metric = mean composite.

### Harness run loop

```
for question in eval_set:
    state = run_graph(question.query, replay_mode=True)
    checks = [tier1, tier2, tier3, tier4, tier5]
    record(question.id, checks, state.meta, checkpointer_ref)
report = render_report(results, baseline=last_run)
```

Each question's full LangGraph checkpointer state is persisted with the run ID — drill-down to "what the graph saw at each node" is one click.

### Reporting

- **HTML dashboard** per run: per-tier scores, per-chapter breakdown, regression diff vs baseline, checkpointer pointer for failures.
- **Baseline composite** in `eval/baselines.json`, committed. PR drops below baseline → CI fails.
- **Per-question timeline** via `eval_runs` Postgres table.
- **Failure triage report** — new vs legacy failures separated.

### Run triggers

- **Local:** `eval/run.py --set ibadat-v1` full run (~20 min with caching).
- **CI:** smoke subset on feature branches; full set on PR-to-main.
- **Nightly on main:** full run, appended to baseline history.

**Cost:** 75 queries × ~25K input × cached + ~3K output ≈ $0.50 per full run.

---

## 9. Error handling, observability, testing

### Per-node failure policies

| Failure mode | Policy |
|---|---|
| **A** — classifier returns invalid/unparseable scope | Retry once with stricter prompt; 2nd failure → refuse as `classification_failed`. |
| **B** — Pinecone timeout on one namespace | Proceed with partial retrieval; mark `meta.partial_retrieval`. If ≥2 namespaces fail → refuse as `retrieval_unavailable`. |
| **B** — rerank call fails | Skip rerank; use raw vector top-k with lower confidence flag. |
| **B** — zero results in a required namespace | Not an error. C catches as `insufficient`, D retriggers. |
| **C** — malformed sufficiency JSON | Retry once; 2nd failure → treat as sufficient by default, flag `meta.sufficiency_check_skipped`. |
| **D** — iteration cap hit without sufficiency | Proceed to E with `sufficiency.capped: true`. |
| **F** — one of 4 parallel synthesis calls fails | Retry that madhab only (up to 2x). Persistent failure → `degraded: true` for that position, others ship. |
| **F** — Citations API returns without anchors | Hard failure for that madhab; retry once; then degraded mode. |
| **V** — citation audit | Three-tier escalation (§7). |
| **G** — cross-madhab analysis fails | Non-blocking. Ship positions without `cross_madhab_analysis`; flag `meta.cross_madhab_degraded`. |
| **H** — narrative rendering fails | Non-blocking. Ship JSON without `narrative`. |

**Principle:** per-madhab fan-outs fail independently; cross-cutting nodes (A, B, V) fail loudly; synthesis-layer nodes (G, H) fail gracefully.

### Observability stack

- **Trace key:** `query_id` through every log line, checkpointer row, trace row, eval result.
- **Structured JSON logs** to stdout (platform logs).
- **LangGraph checkpointer** (Postgres) — primary debugging artifact.
- **Retrieval trace rows** (Postgres) — separate from checkpointer for retrieval-specific detail.
- **Metrics** exposed at `/metrics`:
  - `bayan_query_total{scope}`
  - `bayan_query_latency_ms{node}`
  - `bayan_cache_hit_rate{prompt}`
  - `bayan_citation_audit_result{outcome}`
  - `bayan_sufficiency_iterations{result}`
  - `bayan_refusal_total{reason}`

**MVP observability:** stdout logs + `make observability-report` script querying Postgres. Full Prometheus/Grafana is Phase 1.5.

### Testing strategy — layered

1. **Node unit tests.** One test file per node; LLM + Pinecone mocked at client boundary. Happy path + each documented failure mode + schema contract.
2. **Sub-graph integration tests.** Node groups wired against fixture Pinecone + mocked LLM.
3. **End-to-end tests.** Full graph + fixture corpus + recorded-LLM-responses (VCR-style) for 5-8 canonical queries.
4. **Ingestion tests.** Fixture source texts → expected chunks, metadata, role tags, normalization scores.
5. **Prompt regression tests.** Snapshot cached system prompts; diffs visible in PR.
6. **Contract tests.** Pydantic models validate representative inputs, reject malformed.
7. **Eval harness.** §8. CI gate on composite regression.

**TDD discipline:** every node starts with tests. `test_classify_query.py` exists before `classify_query.py` has a body.

### Operational targets (yardsticks, not SLAs)

- p50 latency ≤ 12s; p95 ≤ 22s
- Refusal latency ≤ 1.5s
- Cost per in-scope query (post-caching) ≤ $0.04
- Cache hit rate on synthesis system prompt ≥ 85%
- Citation audit first-pass rate ≥ 95%
- Eval composite on ibadat-v1 for MVP launch ≥ 0.75 (baselined)

### Refusal behavior

Refusals are first-class product behavior:
- `refusal_reason` — controlled vocab: `out_of_scope_fiqh`, `out_of_scope_non_fiqh`, `contemporary`, `adversarial`, `nonsense`, `classification_failed`, `retrieval_unavailable`, `citation_audit_failed`
- `refusal_detail` — short prose explaining what the system will not answer
- `refusal_suggestion` — if applicable, rephrasing guidance or scholar referral
- UI renders as a distinct panel, not a generic error

---

## 10. Published skills

Three standalone Claude Code skills, published independently at MVP launch. Reusable domain IP separate from product glue.

### Skill 1 — `bayan-arabic-normalization`

**Purpose:** Deterministic Arabic text normalization for embedding/retrieval pipelines.

**Public surface:**
- `normalize(text, profile=...) → (normalized_text, quality_score, flags)`
- Profiles: `embedding` (aggressive), `display` (identity), `comparison` (moderate).

**Scope:** classical/MSA only; not a transliterator; not a morphological analyzer; no dialectal coverage.

### Skill 2 — `bayan-madhab-citation-hierarchy`

**Purpose:** Evidence hierarchy and citation validation for Islamic-knowledge RAG.

**Public surface:**
- `rank_evidence(evidence_list) → ranked_list`
- `validate_hierarchy(position_json) → (valid: bool, violations: [...])`
- Controlled vocabularies: `EvidenceType`, `EvidenceRole`, `AuthorityLevel`, `RoleInSource`, `TawaturStatus`, `IkhtilafReasonType`.
- JSON schema for `Position` / `IkhtilafReason` structures.

**Scope:** rules + schemas only. Not a fiqh reasoner; no corpus dependencies; madhab-agnostic mechanics.

### Skill 3 — `bayan-hadith-chain-handling`

**Purpose:** Hadith matn/chain/cluster data model — per-occurrence chunks, matn clustering, tawatur classification, grader disagreement, cross-collection dedup.

**Public surface:**
- Pydantic models: `HadithOccurrence`, `MatnCluster`, `ChainSummary`, `ChainGrade`, `GraderDispute`.
- `cluster_matns(occurrences) → dict[cluster_id, MatnCluster]`
- `dedupe_retrieval(occurrences) → list[HadithOccurrence]`
- `import_tawatur_flags(path) → dict[cluster_id, TawaturStatus]`
- `effective_grade(grades) → Grade` — conservative-grade resolver.
- Bundled reference data: parsed mutawatir lists (al-Suyuti, al-Kattani).

**Scope:** data modeling + helpers. Not ilm al-rijal; no narrator biography; no chain-independence derivation; Kutub al-Sittah only.

### Release flow

- **During MVP:** all skills in `bayan-graph/skills/` (monorepo), editable install.
- **At MVP launch:** `git subtree split` per skill to standalone GitHub repos; tag `v0.1.0`; publish to Claude Code skills registry; main codebase pins published versions.
- **Versioning:** semver. Breaking vocab/schema = major; new helpers/entries = minor; fixes = patch.

### What is not a published skill

LangGraph nodes (product-coupled), eval harness (ground truth is project-specific; methodology documented instead), FastAPI/Next.js app (commodity glue), standalone ikhtilaf-taxonomy skill (folded into skill 2).

---

## 11. Phase 1.5 roadmap (not in MVP; explicit)

Named here so they don't creep into Phase 1:

- Muamalat expansion (commerce, marriage, inheritance) — new corpus, new eval set.
- Arabic query input — Arabic normalization skill extends to query-side; dialect robustness.
- Streaming responses at the API and UI layer.
- Multi-turn / conversational memory.
- Narrator-level isnad graphing (Neo4j) and chain-independence derivation — this is where "graph" genuinely earns its name.
- Ilm al-rijal integration — Tahdhib al-Kamal, Mizan al-I'tidal.
- IslamicMMLU subset as a secondary eval signal.
- LLM-as-judge for prose narrative quality.
- Full Prometheus/Grafana observability stack.
- Hadith outside Kutub al-Sittah — Ibn Hibban, Ibn Khuzaymah, Mustadrak al-Hakim.
- Community contribution pipeline for eval set (CONTRIBUTING_EVAL.md, review workflow).

---

## 12. Open questions

None blocking the plan. Every decision above is settled. Questions explicitly deferred to Phase 1.5 are captured in §11.
