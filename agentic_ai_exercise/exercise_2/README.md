## Exercise 2: Memory

Check the other README.md's for more information about the exercise structure. For exercise 2, issues are under the Milestone 2.

---

## Setup

### Prerequisites

#### 1. Remote API access

The code reads credentials from a `.env` file at the repo root. Create it if it does not exist:

```
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=<api-base-url>
```

Connect to the techfak VPN before running.

#### 2. Python environment

Requires **Python ≥ 3.13**. The environment is managed with `uv`.

Create and activate a dedicated virtual environment inside `exercise_2/`:

```bash
cd agentic_ai_exercise/exercise_2
uv venv mem0 --python 3.14
source mem0/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
uv pip install -e ../..   # installs the agentic_ai_exercise package
```

### Embedding model

The embedding model is available via the same remote API as the chat model. Example usage:

```python
import os

from dotenv import load_dotenv
from agentscope.embedding import OpenAITextEmbedding
from agentic_ai_exercise import ENV_PATH, QWEN3_06B_Embed, QWEN3_VL_4B_Instruct

# Load API key and base URL from .env
load_dotenv(ENV_PATH)

api_key  = os.environ["LLM_API_KEY"]
api_base = os.environ["LLM_BASE_URL"]

# --- Embedding model (standalone, used outside the agent) ---
embedding_model = OpenAITextEmbedding(
    model_name=QWEN3_06B_Embed,
    api_key=api_key,
    base_url=api_base,           # note: base_url, not client_args
    dimensions=None,
)
```

If you use another class for calling this API, make sure the argument `"encoding_format": "float"` is set. If you want to set the dimensions, you will get an error — if your approach requires a different embedding size, you need a workaround.

### Notes

- `embedding_model_dims` must be set to **1024** in the `VectorStoreConfig` to match `Qwen3-Embedding-0.6B`'s output size. mem0's default is 1536 (OpenAI's size).
- If you get a dimension mismatch error after changing the embedding model, delete the stale local Qdrant collection: `rm -rf ~/.mem0`
- The remote embedding endpoint does not accept the `dimensions` parameter; `OpenAITextEmbedding` is initialised with `dimensions=None` to avoid this.
- `Qwen3-VL-4B-Instruct-FP8` has a ~9,400-token context window. HotpotQA passages can exceed this during mem0's memory-extraction step. The evaluation loop adds passages one at a time and stops before the budget is exceeded — every passage that fits is stored whole, and the rest are skipped.

---

## Running the evaluation

```bash
cd agentic_ai_exercise/exercise_2
source mem0/bin/activate

# HotpotQA — baseline
python3 -m eval.hotpotqa --output results/baseline_hotpot.json --limit 100 --baseline

# HotpotQA — memory agent
python3 -m eval.hotpotqa --output results/memory_hotpot.json --limit 100

# GraphRAG-Bench — baseline (medical subset)
python3 -m eval.graphrag_bench --output results/baseline_graphrag.json --subset medical --limit 100 --baseline

# GraphRAG-Bench — memory agent (medical subset)
python3 -m eval.graphrag_bench --output results/memory_graphrag.json --subset medical --limit 100
```

Use `--limit N` to evaluate on a smaller sample first. Available subsets for GraphRAG-Bench: `medical`, `novel`.

---

## Implementation

### Architecture

The system is built around two components:

**`agent/memory_agent.py` — `MemoryAgent`**

A `MemoryAgent` wraps an `AgentBase` from AgentScope with a `Mem0LongTermMemory` instance. Each `reply()` call follows three steps:

1. **Retrieve** — query the mem0 vector store for memories relevant to the incoming message.
2. **Generate** — build a prompt from an optional system prompt, retrieved memories, and the user message; call the LLM.
3. **Record** — store the Q&A exchange back into mem0 so future questions can retrieve it.

```
User question
     │
     ▼
[mem0 retrieve]  ──→  relevant memories (extracted facts from past exchanges)
     │
     ▼
[LLM call]  ←──  system prompt + memories + question
     │
     ▼
[mem0 record]  ──→  store Q&A exchange for future retrieval
     │
     ▼
Answer
```

**Models used:**
- Chat model: `Qwen/Qwen3-VL-4B-Instruct-FP8`
- Memory extraction model: `Qwen/Qwen3-VL-4B-Instruct-FP8`
- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (1024-dimensional)
- Vector store: Qdrant (in-memory)

**Key design decisions:**

- A **shared agent** is used across all evaluation samples so that memory genuinely accumulates — earlier Q&A exchanges can inform later answers. This is what makes the memory agent meaningfully different from the baseline.
- Context passages are recorded into mem0 individually (not as one blob) so the extraction LLM operates on manageable chunks.
- `reply()` accepts an optional `system_prompt` parameter so the caller can inject instructions (e.g. answer format constraints) without modifying the memory pipeline.
- Retrieve failures are caught and treated as empty memory (non-fatal), ensuring evaluation continues even if the embedding API times out on the first call.

**`agent/memory_agent.py` — `build_chat_model()`**

Also exported for use in the baseline evaluations, which call the same model directly without going through the memory pipeline.

---

### Evaluation

Two benchmarks are evaluated, each with a **baseline** mode (direct context injection) and a **memory agent** mode (mem0 retrieval).

#### Baseline vs. Memory Agent

| Mode | Context delivery | Memory |
|---|---|---|
| **Baseline** | Full context injected into system prompt per sample | None |
| **Memory agent** | No direct context injection | Shared agent; passages recorded into mem0; retrieved by embedding similarity |

This is a genuine comparison: the baseline always sees the exact source text, while the memory agent must retrieve the right facts from an accumulating vector store.

---

#### Dataset 1 — HotpotQA (`eval/hotpotqa.py`)

HotpotQA is a multi-hop QA dataset. Each sample contains a question, a set of Wikipedia passages (distractor setting), and a short gold answer.

**Evaluation protocol:**
- Context passages are truncated to 24,000 characters total.
- For the **baseline**: all passages are concatenated and injected into the system prompt.
- For the **memory agent**: each passage is recorded into the shared mem0 store individually; the agent retrieves relevant memories at query time.
- Qwen3's `<think>...</think>` reasoning blocks are stripped before scoring.
- The model is instructed to answer with a short phrase or single word.

**Metrics:** Exact Match (EM), Token-level F1

**Run:**
```bash
# Baseline
python3 -m eval.hotpotqa --output results/baseline_hotpot.json --limit 100 --baseline

# Memory agent
python3 -m eval.hotpotqa --output results/memory_hotpot.json --limit 100
```

**Results (100 samples, validation split):**

| Mode | Exact Match | F1 |
|---|---|---|
| Baseline | 0.61 | 0.71 |
| Memory agent | **0.63** | **0.73** |


---

#### Dataset 2 — GraphRAG-Bench (`eval/graphrag_bench.py`)

GraphRAG-Bench is a benchmark for Graph RAG systems, covering both medical and novel corpora. Each sample includes a question, a gold evidence passage, a gold answer, and a `question_type` label (levels 1–4: Fact Retrieval, Complex Reasoning, Contextual Summarization, Creative Generation).

**Evaluation protocol:**
- For the **baseline**: the gold evidence is injected directly into the system prompt.
- For the **memory agent**: the evidence is recorded into the shared mem0 store; the agent retrieves relevant memories at query time without direct context injection.
- Results are broken down by `question_type` to capture differences across difficulty levels.

**Metrics:** Exact Match (EM), Token-level F1, ROUGE-L

ROUGE-L is the primary metric for levels 3–4 (open-ended generation), where gold answers are long and EM is not meaningful.

**Run:**
```bash
# Baseline
python3 -m eval.graphrag_bench --output results/baseline_graphrag.json --subset medical --limit 100 --baseline

# Memory agent
python3 -m eval.graphrag_bench --output results/memory_graphrag.json --subset medical --limit 100
```

Available subsets: `medical`, `novel`.

**Results (100 samples, medical subset, Fact Retrieval questions):**

| Mode | Exact Match | F1 | ROUGE-L |
|---|---|---|---|
| Baseline | **0.09** | **0.68** | **0.67** |
| Memory agent | 0.00 | 0.22 | 0.19 |

The baseline strongly outperforms the memory agent here. This reflects a fundamental limitation of mem0 in this setting: its extraction LLM rewrites passages into abstract, profile-style facts (e.g. "Basal cell carcinoma is the most common skin cancer") which can lose precise relational details. For open-domain medical fact retrieval, direct context injection might be a more reliable strategy.

---

### Metrics (`eval/metrics.py`)

All metrics are computed after normalising text (lowercase, punctuation removal, whitespace normalisation).

| Metric | Description |
|---|---|
| **Exact Match** | 1 if normalised prediction equals normalised gold, 0 otherwise |
| **Token F1** | Harmonic mean of precision and recall over shared token counts |
| **ROUGE-L** | F1 based on the Longest Common Subsequence (LCS) of tokens |

---



