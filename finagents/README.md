# finagents

A multi-agent financial reasoning and decision-support system. It answers a
question about a company by planning which lenses are needed, retrieving
evidence for each one, having specialist agents reason independently, verifying
every claim against the cited source, and then synthesising a report in which
each number can be traced back to the document it came from.

The design assumption is that an unsourced financial answer is worse than no
answer. Every claim in the output carries a citation; every citation carries a
retrieval score and the reason it was selected; every confidence number is
decomposed into components a reader can disagree with individually.

Evidence is live. Documents come from SEC EDGAR, Google News and Yahoo Finance;
there is no bundled corpus, and the system never fabricates a document to fill a
gap. A source that fails is reported as a source that failed.

## Architecture

```
                      +-------------+
   UserQuery -------> |   Planner   |  intent, perspective selection, task decomposition
                      +------+------+
                             | tasks
        +--------------------+--------------------+
        v                    v                    v
  +-----------+        +-----------+        +-----------+
  | Financial |        |   News    |        | Technical |   (+ Risk)
  +-----+-----+        +-----+-----+        +-----+-----+
        |                    |                    |
        +--------------------+--------------------+
                             | findings + citations
                      +------+------+
                      |  Retrieval   |  BM25 + embeddings + metadata gate + KG prior,
                      +------+------+  fused by RRF
                             |
                      +------+------+
                      |  Verifier   |  citation integrity, numeric check, entailment
                      +------+------+  -> verdict per claim
                             |
                      +------+------+
                      | Synthesis   |  only verified claims reach the thesis
                      +------+------+
                             |
                     FinancialResearchReport
```

Orchestration is a LangGraph state machine (`finagents/workflow/`), so each
stage is a node that can be inspected, retried or checkpointed independently:

```
plan -> retrieve -> analyse -> verify -> synthesise -> report
                                      \-> fail
```

`analyse` fans out across the specialised agents concurrently; each agent
failing is recorded in `errors` and the run continues, because a memo built
from three lenses beats no memo.

Retrieval is scoped to the securities the query actually resolves to, so a
one-company question fetches one company's filings rather than a fixed corpus.

## Modules

| Module | Responsibility |
| --- | --- |
| `finagents/schemas.py` | Every domain model: query, plan, task, finding, claim, citation, verification, conflict, confidence, report, and `Settings` |
| `finagents/llm/` | `LLMClient` contract, Ollama REST client, provider factory |
| `finagents/data/` | SEC EDGAR, Google News and yfinance clients; the online `MarketDataProvider`; indicator computation |
| `finagents/retrieval/` | BM25, hashing/Ollama embedder, metadata filters, RRF fusion, financial knowledge graph |
| `finagents/agents/` | Planner plus the financial, news, technical and risk agents |
| `finagents/verification/` | Claim verifier, conflict detector, confidence assessor, gated synthesizer |
| `finagents/workflow/` | LangGraph state, nodes, edges, fault policy |
| `finagents/reporting/` | Human-readable and JSON report rendering |
| `finagents/cli.py` | `finagents` entry point |

## Install

Requires Python 3.10+ (developed on 3.12).

```bash
cd finagents
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
```

### SEC requires a declared identity

SEC fair-access policy asks callers to identify themselves and throttles or
blocks anonymous bulk traffic. Set a real contact before the first run:

```bash
# PowerShell
$env:FINAGENTS_SEC_USER_AGENT = "Jane Doe jane@example.com"
# bash
export FINAGENTS_SEC_USER_AGENT="Jane Doe jane@example.com"
```

The default is a placeholder. It logs a warning at startup and the run will get
throttled or refused.

The client also stays under EDGAR's 10 requests/second limit (0.34s between
requests) and caches responses under `.cache/finagents/sec`, so repeat questions
about the same company cost nothing.

### Enabling model reasoning

Ollama is the only LLM. There are no API keys and no cloud calls.

```bash
# 1. Install Ollama from https://ollama.com/download
ollama serve
# 2. Pull the models
ollama pull llama3.1
ollama pull nomic-embed-text     # optional: real embeddings instead of hashing
```

Check the wiring at any time:

```bash
finagents --doctor
```

If Ollama is not reachable, `build_llm_client` raises. This is deliberate: a
system that silently downgrades to canned text while reporting a real-looking
report is worse than one that refuses to start.

### Expect it to be slow without a GPU

Ollama prefills prompts at roughly 3ms per character on CPU. A 9,000-character
evidence block — six retrieved documents — takes about 80 seconds before the
first token, and a full four-agent run takes several minutes. This is the hardware,
not the code. On a GPU the same prompt returns in a few seconds, and you can drop
`FINAGENTS_LLM_TIMEOUT_S` accordingly.

A run where one agent fails still produces a report; the failure is recorded in
the confidence penalties rather than hidden.

## Usage

```bash
# Research a question
finagents "What are the main risks facing NVDA?"

# Force a ticker and get machine-readable output
finagents --json --ticker NVDA "Is the valuation justified?"

# Sequential agents, no verification layer, wider retrieval
finagents --sequential --no-verify --top-k 12 "Analyse AAPL fundamentals"

# Provider health
finagents --doctor
```

Exit codes: `0` success, `1` failure, `2` no query, `3` the run completed but
confidence fell below the floor.

Ticker symbols are validated against EDGAR's filing index, so class and
preferred shares work: `BRK-B`, `BRK.B` and `BRK B` all resolve to Berkshire
class B, and an invented symbol is dropped before retrieval rather than after.

### As a library

```python
from finagents.schemas import Settings, UserQuery
from finagents.workflow import FinancialResearchWorkflow
from finagents.reporting import render

settings = Settings(sec_user_agent="Jane Doe jane@example.com")
workflow = FinancialResearchWorkflow(settings)
state = workflow.run(UserQuery(raw_text="What are the main risks facing NVDA?"))

print(render(state.report))
print(state.report.overall_stance, state.report.confidence.overall)
```

## Configuration

Settings are read from the environment, prefix `FINAGENTS_`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FINAGENTS_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `FINAGENTS_OLLAMA_MODEL` | `llama3.1` | reasoning model |
| `FINAGENTS_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | embedding model |
| `FINAGENTS_LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `FINAGENTS_LLM_MAX_TOKENS` | `2048` | response cap |
| `FINAGENTS_LLM_TIMEOUT_S` | `300` | request timeout; see the note below |
| `FINAGENTS_SEC_USER_AGENT` | placeholder | SEC contact, e.g. `Jane Doe jane@example.com` |
| `FINAGENTS_CACHE_DIR` | `.cache/finagents` | on-disk response cache |
| `FINAGENTS_FILINGS_PER_TICKER` | `3` | recent filings fetched per security |
| `FINAGENTS_MAX_CHUNKS_PER_FILING` | `20` | retrievable chunks kept per filing |
| `FINAGENTS_MAX_CHUNKS_PER_SECTION` | `3` | chunks kept per filing section |
| `FINAGENTS_NEWS_PER_TICKER` | `10` | Google News items per security |
| `FINAGENTS_NEWS_MIN_BODY_CHARS` | `120` | drop headline-only items |
| `FINAGENTS_HTTP_TIMEOUT_S` | `20` | data-source request timeout |
| `FINAGENTS_TOP_K` | `8` | documents per single search |
| `FINAGENTS_TOP_K_PER_TASK` | `6` | documents per planner task |
| `FINAGENTS_KG_MAX_HOPS` | `2` | knowledge-graph traversal depth |
| `FINAGENTS_PARALLEL` | `true` | run agents concurrently |
| `FINAGENTS_VERIFY` | `true` | run the verification layer |

## Data sources

| Source | Used for | Reliability |
| --- | --- | --- |
| SEC EDGAR submissions | filings list, dates, accession numbers | 1.0 |
| SEC EDGAR inline XBRL | audited figures, tagged by the filer | 1.0 |
| SEC EDGAR filing text | risk factors, MD&A, business description | form-dependent |
| Google News RSS | headlines and lead paragraphs | outlet-weighted |
| Yahoo Finance (yfinance) | one year of OHLCV for indicators | 0.5 |

Every document records its source, a retrieval score, and why it was selected.
The rendered report ends with a `DATA SOURCES` section listing what was actually
consulted.

Two consequences worth being explicit about:

- **News is an aggregator.** Google News surfaces outlets, not articles. Only
  the headline and lead paragraph are retrievable, and provenance is recorded as
  `Reuters (via Google News)` so a reader is never misled about the original.
- **Figures come from XBRL, prose comes from the filing.** The financial
  statements are tables; the narrative is the parts worth reading. The extractor
  drops table-like chunks and budgets retrieval across sections, so
  "Item 1A. Risk Factors" is reachable instead of being crowded out by
  balance-sheet rows.

## Tests

```bash
pytest              # 211 tests, ~30s
```

The suite runs entirely offline: no network, no model daemon, no keys. That is a
constraint, not a convenience — a test suite for a financial system that quietly
depends on a local daemon is a suite that stops being runnable the moment the
daemon is not up. The network clients are tested against fixture tables that
reproduce the awkward parts of real payloads (the hidden XBRL header, a 10-Q
whose Item 1 is 30x longer than its risk factors, `BRK-B`).

The adversarial cases are the interesting ones: a claim citing a figure its own
source does not contain, a claim citing a document that was never retrieved, an
uncited claim, a duplicated claim, a neutral lens, and an agent that raises
mid-run. Each is expected to be caught, not smoothed over.

## Limitations

- Requires Ollama. There is no built-in reasoning fallback.
- Google News gives headlines and lead paragraphs, not full articles.
- The knowledge graph is derived from retrieved documents, so coverage is bounded
  by what was fetched.
- No backtesting: nothing here is evidence that the system produces profitable
  recommendations.
- Not investment advice.
