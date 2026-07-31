# cosmos-agent-memory-lab

A runnable companion to the *Cosmos DB for Agentic AI* series on
[Cloud Perspectives](https://sjwiggers.com). Every module maps to a
specific post and implements the code shown there against a real Cosmos DB
SDK, backed by a test suite that runs with zero Azure account required.

| Post | Topic | Module(s) |
|---|---|---|
| 1 | Why Cosmos DB for agent memory | — (framing post, no code) |
| 2 | Designing a Cosmos DB agent memory schema | `schema.py`, `models.py`, `seed.py` |
| 3 | Vector, full-text, and hybrid search | `search.py` |
| 4 | Multi-agent state and checkpointing | `checkpointer.py`, `graph.py`, `change_feed_demo.py`, `functions/change_feed_handoff/` |
| 5 | Foundry Agent Service (upcoming) | — |
| 6 | Cost, caching, and pitfalls (upcoming) | — |

## What's actually in here

- **A hierarchical-partition-key, TTL-enabled schema** (`schema.py`) for the
  turn-based memory item from post 2, with a vector + full-text indexing
  policy for post 3 that gets skipped automatically when it detects it's
  running against the Cosmos DB Emulator (which doesn't support vector
  policies yet).
- **The four search patterns from post 3** (`search.py`) as testable
  query-builder functions -- recency, semantic, hybrid (RRF), and keyword.
- **A from-scratch Cosmos DB-backed LangGraph checkpointer** (`checkpointer.py`)
  implementing `BaseCheckpointSaver` directly against the Cosmos Python SDK,
  rather than wrapping a third-party package -- the whole persistence path
  is one file, and it's exercised by real LangGraph runs in the test suite.
- **A minimal triage → specialist multi-agent graph** (`graph.py`) in the
  shape of Microsoft's own
  [multi-agent-langgraph](https://github.com/AzureCosmosDB/multi-agent-langgraph)
  sample, trimmed down to the essentials.
- **Change feed as the handoff mechanism** (`change_feed_demo.py`), runnable
  locally with the Cosmos SDK's change feed API, plus a deployable Azure
  Function version (`functions/change_feed_handoff/`) using the same logic
  with a Cosmos DB trigger binding.

## Quickstart

### Option A: local, no Azure account (posts 2 and 4 only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

docker compose up -d          # starts the Cosmos DB Emulator
python scripts/provision.py   # creates the database + containers
python scripts/seed_data.py   # seeds sample turns
python scripts/run_demo.py    # runs schema + checkpointing + change feed end to end
```

Post 3's vector and hybrid queries are skipped automatically in this mode --
the Linux Cosmos DB Emulator doesn't yet support vector indexing policies.

### Option B: real Azure Cosmos DB account (all of posts 2-4, including vector search)

```bash
cp .env.example .env          # fill in COSMOS_ENDPOINT and COSMOS_KEY
export $(cat .env | xargs)    # or use your preferred env loader
python scripts/provision.py
python scripts/seed_data.py
python scripts/run_demo.py
```

Enable the **Vector Search** and **Full-Text Search** preview features on
your Cosmos DB for NoSQL account before running `provision.py`, or container
creation will fail on the vector/full-text policy.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

No Cosmos DB account, emulator, or Docker required -- the test suite runs
against an in-memory fake of the Cosmos container API
(`tests/conftest.py`), including real end-to-end LangGraph runs through
`checkpointer.CosmosDBSaver`. This is also what CI runs on every push (see
`.github/workflows/ci.yml`).

## Project layout

```
src/cosmos_agent_lab/
  config.py            Emulator vs. real-account client setup
  models.py             Turn-based memory item (post 2)
  schema.py              Container provisioning: partition keys, TTL, vector + full-text policy (post 2)
  seed.py                 Sample data with deterministic fake embeddings
  search.py                Four query patterns (post 3)
  checkpointer.py            Cosmos-backed LangGraph checkpoint saver (post 4)
  graph.py                    Triage -> specialist multi-agent graph (post 4)
  change_feed_demo.py          Local change-feed handoff reader (post 4)
functions/change_feed_handoff/  Deployable Azure Function version of the same handoff logic
scripts/                         CLI entry points (provision, seed, full demo)
tests/                            pytest suite + the in-memory fake Cosmos container
```

## Design notes worth knowing before you extend this

- **Encoding `tenantId:threadId` into `thread_id`.** LangGraph's
  `thread_id` is a single string; this repo encodes the tenant into it
  (`"contoso:thread-1234"`) so the checkpointer's hierarchical partition
  key lines up with the `turns` container's partition key from post 2. A
  bare `thread_id` with no `:` falls back to a `"default"` tenant.
- **The checkpointer stores whole checkpoints, not delta blobs.** LangGraph's
  built-in `InMemorySaver` splits channel values into separately versioned
  blobs for compaction. This implementation stores each checkpoint as one
  self-contained item instead -- simpler, easier to reason about in a
  teaching sample, and the same approach most third-party checkpointers
  (Postgres, SQLite) actually use.
- **Fake embeddings.** `seed.py`'s `fake_embedding()` hashes text into a
  deterministic vector so the demo needs no embeddings API key. Swap in a
  real embedding model before using this schema for anything beyond
  learning the patterns.
- **Embedding dimensions are small on purpose** (`schema.EMBEDDING_DIMENSIONS = 8`).
  Set this to match your real embedding model before using the schema for
  anything beyond this demo.

## License

MIT -- see [LICENSE](LICENSE). Written to accompany
[sjwiggers.com](https://sjwiggers.com); link back if you build on it, but
you don't have to ask.
