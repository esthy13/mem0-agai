import argparse
import asyncio
import json
import re
from collections import defaultdict

from datasets import load_dataset
from agentscope.message import Msg
from agent.memory_agent import MemoryAgent, build_chat_model
from eval.metrics import f1_score, exact_match, rouge_l

DATASET_ID = "GraphRAG-Bench/GraphRAG-Bench"
MAX_CONTEXT_CHARS = 24_000
ANSWER_SYSTEM_PROMPT = (
    "Answer the question as accurately and concisely as possible based on the context. "
    "For factual questions, give a short precise answer. "
    "For summarization or creative questions, be thorough but grounded in the context."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str, help="Path to save results.")
    parser.add_argument('--subset', default='medical', choices=['medical', 'novel'],
                        help="Dataset subset to evaluate on (default: medical).")
    parser.add_argument('--limit', type=int, help="Max number of samples to evaluate.")
    parser.add_argument('--baseline', action='store_true',
                        help="Run baseline evaluation (no memory; evidence injected directly into prompt).")
    return parser.parse_args()


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models.

    Args:
        text: Raw model output, potentially containing thinking blocks.

    Returns:
        Text with all thinking blocks removed and whitespace stripped.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def truncate_context(text: str) -> str:
    """Truncate text to MAX_CONTEXT_CHARS if it exceeds the limit.

    Args:
        text: Input context string.

    Returns:
        Truncated or original string.
    """
    return text[:MAX_CONTEXT_CHARS] if len(text) > MAX_CONTEXT_CHARS else text


def score_sample(pred: str, gold: str) -> dict:
    """Compute all evaluation metrics for a single prediction.

    Args:
        pred: Predicted answer string.
        gold: Gold answer string.

    Returns:
        Dict with keys 'exact_match', 'f1', and 'rouge_l'.
    """
    return {
        "exact_match": exact_match(pred, gold),
        "f1": f1_score(pred, gold),
        "rouge_l": rouge_l(pred, gold),
    }


def aggregate(scores: list[dict]) -> dict:
    """Average a list of per-sample score dicts.

    Args:
        scores: List of dicts, each containing 'exact_match', 'f1', 'rouge_l'.

    Returns:
        Dict of averaged scores plus a 'count' field. Returns zeros if the
        list is empty.
    """
    n = len(scores)
    if n == 0:
        return {"exact_match": 0, "f1": 0, "rouge_l": 0, "count": 0}
    return {
        "exact_match": round(sum(s["exact_match"] for s in scores) / n, 4),
        "f1": round(sum(s["f1"] for s in scores) / n, 4),
        "rouge_l": round(sum(s["rouge_l"] for s in scores) / n, 4),
        "count": n,
    }


_shared_agent: MemoryAgent | None = None

async def get_shared_agent() -> MemoryAgent:
    """Return the module-level shared MemoryAgent, creating it on first call.

    Returns:
        Shared MemoryAgent instance.
    """
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = MemoryAgent(name='Memory')
    return _shared_agent


async def run_memory_agent(sample: dict) -> str:
    """Answer a single sample using the shared memory agent.

    The gold evidence is recorded into mem0 before the question is asked, so
    facts accumulate across samples and can be retrieved for later questions.
    No evidence is injected directly into the prompt.

    Args:
        sample: A single GraphRAG-Bench dataset row with keys 'evidence'
            and 'question'.

    Returns:
        Predicted answer string with thinking blocks stripped.
    """
    agent = await get_shared_agent()
    evidence = truncate_context(sample['evidence'] if isinstance(sample['evidence'], str)
                                else " ".join(sample['evidence']))
    # Record evidence into memory so it can be retrieved for future questions (errors are non-fatal)
    try:
        await agent.memory.record([Msg(name="user", role="user", content=evidence)])
    except Exception:
        pass
    # Answer using only retrieved memories, no direct context injection
    response = await agent.reply(Msg("user", sample['question'], "user"),
                                 system_prompt=ANSWER_SYSTEM_PROMPT)
    return strip_thinking(response.get_text_content())


async def run_baseline(model, sample: dict) -> str:
    """Answer a single sample using a plain LLM with evidence in the prompt.

    Args:
        model: Instantiated chat model to call.
        sample: A single GraphRAG-Bench dataset row with keys 'evidence'
            and 'question'.

    Returns:
        Predicted answer string with thinking blocks stripped.
    """
    evidence = truncate_context(sample['evidence'] if isinstance(sample['evidence'], str)
                                else " ".join(sample['evidence']))
    messages = [
        {'role': 'system', 'content': (
            f"Use the following context to answer the question.\n\n{evidence}\n\n"
            f"{ANSWER_SYSTEM_PROMPT}"
        )},
        {'role': 'user', 'content': sample['question']},
    ]
    response = await model(messages)
    raw_text = " ".join(
        block["text"] for block in response.content
        if block.get("type") == "text"
    )
    return strip_thinking(raw_text)


async def evaluate(dataset, output_path: str, baseline: bool = False) -> None:
    """Run the selected evaluation mode on GraphRAG-Bench and write results.

    Results are broken down by question_type (Fact Retrieval, Complex
    Reasoning, Contextual Summarization, Creative Generation) in addition
    to an overall aggregate.

    Args:
        dataset: HuggingFace dataset split to evaluate on.
        output_path: Path to the JSON file where results will be saved.
        baseline: If True, run the plain-LLM baseline; otherwise run the
            shared memory agent.
    """
    mode = "baseline" if baseline else "memory_agent"
    model = build_chat_model() if baseline else None

    by_type: dict[str, list[dict]] = defaultdict(list)
    all_scores: list[dict] = []

    for idx, sample in enumerate(dataset):
        qtype = sample.get('question_type', 'unknown')
        print(f"[{mode}] Sample {idx} | type={qtype}")
        print(f"  Question: {sample['question']}")

        if baseline:
            pred = await run_baseline(model, sample)
        else:
            pred = await run_memory_agent(sample)

        gold = sample['answer']
        scores = score_sample(pred, gold)
        all_scores.append(scores)
        by_type[qtype].append(scores)

        print(f"  Gold     : {gold}")
        print(f"  Predicted: {pred!r}")
        print(f"  EM={scores['exact_match']}  F1={scores['f1']:.3f}  ROUGE-L={scores['rouge_l']:.3f}")

    results = {
        "mode": mode,
        "subset": dataset.config_name if hasattr(dataset, 'config_name') else "unknown",
        "samples_evaluated": len(all_scores),
        "overall": aggregate(all_scores),
        "by_question_type": {qt: aggregate(sc) for qt, sc in sorted(by_type.items())},
    }

    print(f"\nMode: {results['mode']}  |  Samples: {results['samples_evaluated']}")
    print(f"Overall  EM={results['overall']['exact_match']}  "
          f"F1={results['overall']['f1']}  ROUGE-L={results['overall']['rouge_l']}")
    for qt, agg in results['by_question_type'].items():
        print(f"  {qt:30s}  n={agg['count']}  "
              f"EM={agg['exact_match']}  F1={agg['f1']}  ROUGE-L={agg['rouge_l']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    ds = load_dataset(DATASET_ID, args.subset, split="train")
    dataset = ds.select(range(args.limit)) if args.limit else ds
    asyncio.run(evaluate(dataset=dataset, output_path=args.output, baseline=args.baseline))
