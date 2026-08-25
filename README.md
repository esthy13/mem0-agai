# Mem0 Agent Memory Benchmark

An experimental evaluation of **long-term memory for LLM agents**, built with AgentScope, Mem0, Qdrant, and Qwen models. The project compares retrieval-backed memory against direct context injection on multi-hop and knowledge-intensive question answering.

The key finding is practical rather than promotional: memory improved slightly on HotpotQA, while direct context injection remained substantially stronger on medical GraphRAG-Bench questions. This repository documents both outcomes and the engineering trade-offs behind them.

## Highlights

- Built an asynchronous retrieve → generate → record agent pipeline with AgentScope and Mem0.
- Configured Qdrant-backed semantic retrieval with 1,024-dimensional Qwen embeddings.
- Designed controlled baseline-versus-memory evaluations on HotpotQA and GraphRAG-Bench.
- Implemented Exact Match, token F1, and ROUGE-L scoring with per-question-type aggregation.
- Added resilience for embedding/retrieval failures and context-window constraints.

## Results

| Benchmark | Mode | Exact Match | F1 | ROUGE-L |
|---|---|---:|---:|---:|
| HotpotQA, 100 samples | Direct-context baseline | 0.61 | 0.71 | — |
| HotpotQA, 100 samples | Memory agent | **0.63** | **0.73** | — |
| GraphRAG-Bench medical, 100 samples | Direct-context baseline | **0.09** | **0.68** | **0.67** |
| GraphRAG-Bench medical, 100 samples | Memory agent | 0.00 | 0.22 | 0.19 |

The HotpotQA run suggests that accumulated memory can help when related facts recur. On GraphRAG-Bench, Mem0's fact extraction lost relational detail that the baseline retained by seeing the source evidence directly. The result highlights when memory compression is useful—and when retrieval should preserve richer source context.

## Architecture

```text
question ──> semantic memory retrieval ──> Qwen answer generation ──> response
                         ^                                      |
                         └────── record question + answer ──────┘
```

The baseline bypasses retrieval and places the complete evidence directly in the model prompt. Both modes use the same chat model so the comparison isolates the context-delivery strategy.

### How the memory agent works

[`MemoryAgent`](exercise_2/agent/memory_agent.py) wraps an AgentScope `AgentBase` with `Mem0LongTermMemory`. Each request follows three steps:

1. **Retrieve** memories related to the incoming question from Qdrant.
2. **Generate** an answer from the optional system prompt, retrieved memories, and question.
3. **Record** the question-and-answer exchange for later retrieval.

The evaluation uses one shared agent across samples so knowledge genuinely accumulates. Context passages are recorded individually to keep memory extraction within the model's context window. Retrieval and recording failures are non-fatal, allowing long benchmark runs to continue through transient API errors.

The exported `build_chat_model()` factory is also used by the baseline, ensuring both modes use the same generation model.

## Tech stack

- Python 3.13+
- AgentScope and Mem0
- Qdrant vector storage
- Qwen3-VL-4B-Instruct-FP8
- Qwen3-Embedding-0.6B
- Hugging Face Datasets
- pytest, Ruff, and GitLab CI

## Repository layout

```text
.
├── exercise_2/
│   ├── agent/memory_agent.py    # memory agent and model factories
│   ├── eval/                    # benchmark runners and metrics
│   └── results/                 # committed benchmark summaries
├── tests/
├── example.py                   # minimal chat-agent example
├── embedding_example.py         # embedding API example
└── project_config.py            # shared paths and model identifiers
```

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the environment and install the project:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[dev]'
uv pip install -r exercise_2/requirements.txt
```

Create `.env` in the repository root:

```dotenv
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=<openai-compatible-api-base-url>
```

Do not commit credentials.

### Model and embedding configuration

The chat and memory-extraction model is `Qwen/Qwen3-VL-4B-Instruct-FP8`. Semantic retrieval uses `Qwen/Qwen3-Embedding-0.6B` with an in-memory Qdrant vector store.

Important implementation details:

- Set `embedding_model_dims` to `1024`; Mem0's OpenAI-oriented default is `1536`.
- The remote embedding endpoint does not accept a `dimensions` parameter, so the client uses `dimensions=None`.
- Alternative embedding clients must request `encoding_format="float"`.
- HotpotQA context is capped at 24,000 characters to stay within the chat model's context window.
- If a local Qdrant collection was created with a different embedding size, remove that stale collection before rerunning the evaluation.

## Run the evaluations

Run commands from the repository root:

```bash
# HotpotQA
python -m exercise_2.eval.hotpotqa --output exercise_2/results/baseline_hotpot.json --limit 100 --baseline
python -m exercise_2.eval.hotpotqa --output exercise_2/results/memory_hotpot.json --limit 100

# GraphRAG-Bench, medical subset
python -m exercise_2.eval.graphrag_bench --output exercise_2/results/baseline_graphrag.json --subset medical --limit 100 --baseline
python -m exercise_2.eval.graphrag_bench --output exercise_2/results/memory_graphrag.json --subset medical --limit 100
```

Use a small `--limit` first because evaluation calls a remote model endpoint. The GraphRAG-Bench runner also supports `--subset novel`.

## Evaluation methodology

| Mode | Context delivery | Memory behavior |
|---|---|---|
| Direct-context baseline | Source evidence is placed directly in the system prompt | No persistent memory |
| Memory agent | Evidence is recorded, extracted, and retrieved by embedding similarity | Shared memory accumulates across samples |

### HotpotQA

HotpotQA tests multi-hop question answering over Wikipedia passages in the distractor setting.

- The baseline receives all passages that fit within the context cap.
- The memory agent records each passage separately and answers using retrieved facts.
- Qwen `<think>...</think>` blocks are removed before scoring.
- Answers are constrained to a short phrase or single word.
- Metrics: Exact Match and token-level F1.

### GraphRAG-Bench

GraphRAG-Bench evaluates fact retrieval, complex reasoning, contextual summarization, and creative generation over medical and novel corpora.

- The baseline receives the gold evidence directly.
- The memory agent records the evidence and must retrieve it without direct prompt injection.
- Scores are aggregated overall and by question type.
- Metrics: Exact Match, token-level F1, and ROUGE-L.

ROUGE-L is the most informative metric for longer summarization and generation answers, where exact string matching is overly strict.

### Metric definitions

All predictions and references are lowercased and normalized for punctuation and whitespace before scoring.

| Metric | Definition |
|---|---|
| Exact Match | `1` when normalized prediction and reference are identical; otherwise `0` |
| Token F1 | Harmonic mean of precision and recall over shared token counts |
| ROUGE-L | F1 derived from the longest common token subsequence |

## Interpretation and limitations

The small HotpotQA improvement shows that persistent memory can help when relevant facts can be compressed and recovered reliably. The GraphRAG-Bench result exposes the opposite case: Mem0 may rewrite evidence into generalized facts and lose precise relationships needed for medical questions.

This experiment therefore does not claim that memory universally improves retrieval-augmented generation. It demonstrates that the quality of memory extraction and the fidelity of stored evidence are as important as semantic retrieval itself.

## Quality checks

```bash
uv run pytest
uv run ruff check .
```

## Repository metadata

**About:** Benchmarking long-term memory retrieval for LLM agents with Mem0, AgentScope, Qdrant, and Qwen.

**Topics:** `llm-agents`, `agent-memory`, `mem0`, `agentscope`, `qdrant`, `retrieval-augmented-generation`, `rag`, `qwen`, `hotpotqa`, `graphrag`, `benchmarking`, `python`
