import string
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Args:
        text: Raw string to normalise.

    Returns:
        Normalised string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def rouge_l(pred: str, gold: str) -> float:
    """Compute sentence-level ROUGE-L F1 via longest common subsequence.

    Args:
        pred: Predicted answer string.
        gold: Gold answer string.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalised prediction equals normalised gold, else 0.

    Args:
        pred: Predicted answer string.
        gold: Gold answer string.

    Returns:
        1 if strings match after normalisation, 0 otherwise.
    """
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold.

    Args:
        pred: Predicted answer string.
        gold: Gold answer string.

    Returns:
        Token-level F1 score in [0, 1].
    """
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)