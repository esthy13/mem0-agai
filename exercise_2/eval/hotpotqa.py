import argparse
import asyncio
import json

from agentscope.message import Msg
from datasets import load_dataset

from exercise_2.agent.memory_agent import MemoryAgent, build_chat_model
from exercise_2.eval.metrics import exact_match, f1_score

DATASET_TO_EVALUATE = "hotpot_qa"
MAX_CONTEXT_CHARS = 24_000
ANSWER_SYSTEM_PROMPT = (
    "Answer the question with a short phrase or a single word. "
    "Do not include explanations or full sentences."
)

ds = load_dataset(DATASET_TO_EVALUATE, "distractor", split="validation")

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models.

    Qwen3 and similar chain-of-thought models wrap their internal reasoning
    in these tags. Stripping them before scoring ensures only the final
    answer is evaluated.

    Args:
        text: Raw model output, potentially containing thinking blocks.

    Returns:
        Text with all thinking blocks removed and whitespace stripped.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with fields: output, limit, baseline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str, help="Path to save results.")
    parser.add_argument('--limit', type=int, help="Max number of samples to evaluate.")
    parser.add_argument('--baseline', action='store_true',
                        help="Run baseline evaluation (no memory; context injected directly into prompt).")
    return parser.parse_args()


async def evaluate_memory_agent(dataset, output_path: str) -> tuple[float, float, int]:
    """Evaluate the memory agent on HotpotQA.

    A single shared MemoryAgent is used across all samples so that knowledge
    accumulates in mem0 over the course of the evaluation. For each sample,
    context passages are recorded into memory individually; the agent then
    retrieves relevant facts at query time without direct context injection.

    Args:
        dataset: HuggingFace dataset split to evaluate on.
        output_path: Unused; kept for API symmetry with evaluate_baseline.

    Returns:
        Tuple of (total_exact_match, total_f1, num_samples).
    """
    total_number_samples = len(dataset)
    total_em = 0.0
    total_f1 = 0.0

    # Shared agent: memory accumulates across samples so retrieved memories
    # from past Q&A exchanges can inform future answers.
    agent = MemoryAgent(name='Memory')

    for idx, sample in enumerate(dataset):
        print(f'Elaborating sample: {idx} on a total of {total_number_samples}')
        passages = []
        char_count = 0
        for title, sents in zip(sample['context']['title'], sample['context']['sentences']):
            passage = f'{title}: {"".join(sents)}'
            if char_count + len(passage) > MAX_CONTEXT_CHARS:
                break
            passages.append(passage)
            char_count += len(passage)

        # Record context passages into memory so they can be retrieved later
        for passage in passages:
            try:
                await agent.memory.record([Msg(name="user", role="user", content=passage)])
            except Exception:
                pass

        # Answer using retrieved memories — no direct context injection
        response = await agent.reply(
            Msg("user", sample['question'], "user"),
            system_prompt=ANSWER_SYSTEM_PROMPT,
        )

        answer_text = strip_thinking(response.get_text_content())
        em = exact_match(answer_text, sample['answer'])
        f1 = f1_score(answer_text, sample['answer'])
        total_em += em
        total_f1 += f1
        print(f"  Question : {sample['question']}")
        print(f"  Gold     : {sample['answer']}")
        print(f"  Predicted: {answer_text!r}")
        print(f"  EM={em}  F1={f1:.3f}")

    return total_em, total_f1, total_number_samples

async def evaluate_baseline(dataset, output_path: str) -> tuple[float, float, int]:
    """Evaluate a plain LLM baseline on HotpotQA.

    For each sample the full concatenated context is injected directly into
    the system prompt. No memory is used.

    Args:
        dataset: HuggingFace dataset split to evaluate on.
        output_path: Unused; kept for API symmetry with evaluate_memory_agent.

    Returns:
        Tuple of (total_exact_match, total_f1, num_samples).
    """
    model = build_chat_model()
    total_number_samples = len(dataset)
    total_em = 0.0
    total_f1 = 0.0

    for idx, sample in enumerate(dataset):
        print(f'[Baseline] Elaborating sample: {idx} on a total of {total_number_samples}')
        passages = []
        char_count = 0
        for title, sents in zip(sample['context']['title'], sample['context']['sentences']):
            passage = f'{title}: {"".join(sents)}'
            if char_count + len(passage) > MAX_CONTEXT_CHARS:
                break
            passages.append(passage)
            char_count += len(passage)
        context_text = "\n".join(passages)

        messages = [
            {'role': 'system', 'content': (
                f"Use the following context to answer the question.\n\n{context_text}\n\n"
                f"{ANSWER_SYSTEM_PROMPT}"
            )},
            {'role': 'user', 'content': sample['question']},
        ]
        response = await model(messages)
        raw_text = " ".join(
            block["text"] for block in response.content
            if block.get("type") == "text"
        )
        answer_text = strip_thinking(raw_text)
        em = exact_match(answer_text, sample['answer'])
        f1 = f1_score(answer_text, sample['answer'])
        total_em += em
        total_f1 += f1
        print(f"  Question : {sample['question']}")
        print(f"  Gold     : {sample['answer']}")
        print(f"  Predicted: {answer_text!r}")
        print(f"  EM={em}  F1={f1:.3f}")

    return total_em, total_f1, total_number_samples

async def evaluate(dataset, output_path: str, baseline: bool = False) -> None:
    """Run the selected evaluation mode and write results to disk.

    Args:
        dataset: HuggingFace dataset split to evaluate on.
        output_path: Path to the JSON file where results will be saved.
        baseline: If True, run the plain-LLM baseline; otherwise run the
            memory agent.
    """
    if baseline:
        total_em, total_f1, n = await evaluate_baseline(dataset, output_path)
        mode = "baseline"
    else:
        total_em, total_f1, n = await evaluate_memory_agent(dataset, output_path)
        mode = "memory_agent"

    results = {
        "mode": mode,
        "samples_evaluated": n,
        "exact_match": round(total_em / n, 4),
        "f1_score": round(total_f1 / n, 4),
    }

    print(f"Mode:             {results['mode']}")
    print(f"Samples evaluated: {results['samples_evaluated']}")
    print(f"Exact Match:       {results['exact_match']}")
    print(f"F1 Score:          {results['f1_score']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    args = parse_args()
    dataset = ds.select(range(args.limit)) if args.limit else ds
    asyncio.run(
        evaluate(dataset=dataset, output_path=args.output, baseline=args.baseline)
    )
    
