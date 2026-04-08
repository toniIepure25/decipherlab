from __future__ import annotations

from decipherlab.decoding.beam_search import BeamDecodingResult


def sequence_exact_match(predicted: list[str], truth: list[str]) -> float:
    return float(predicted == truth)


def sequence_token_accuracy(predicted: list[str], truth: list[str]) -> float:
    if not truth:
        return 0.0
    matches = sum(left == right for left, right in zip(predicted, truth))
    return matches / len(truth)


def sequence_edit_distance(predicted: list[str], truth: list[str]) -> int:
    if not predicted:
        return len(truth)
    if not truth:
        return len(predicted)
    distances = [[0] * (len(truth) + 1) for _ in range(len(predicted) + 1)]
    for index in range(len(predicted) + 1):
        distances[index][0] = index
    for index in range(len(truth) + 1):
        distances[0][index] = index
    for row_index, left in enumerate(predicted, start=1):
        for column_index, right in enumerate(truth, start=1):
            substitution_cost = 0 if left == right else 1
            distances[row_index][column_index] = min(
                distances[row_index - 1][column_index] + 1,
                distances[row_index][column_index - 1] + 1,
                distances[row_index - 1][column_index - 1] + substitution_cost,
            )
    return distances[-1][-1]


def sequence_top_k_recovery(result: BeamDecodingResult, truth: list[str]) -> float:
    truth_tuple = tuple(truth)
    return float(any(tuple(sequence.symbols) == truth_tuple for sequence in result.sequences))


def sequence_metric_bundle(result: BeamDecodingResult, truth: list[str]) -> dict[str, float]:
    predicted = result.best.symbols if result.sequences else []
    edit_distance = sequence_edit_distance(predicted, truth)
    return {
        "sequence_exact_match": sequence_exact_match(predicted, truth),
        "sequence_token_accuracy": sequence_token_accuracy(predicted, truth),
        "sequence_topk_recovery": sequence_top_k_recovery(result, truth),
        "sequence_cer": (edit_distance / len(truth)) if truth else 0.0,
    }
