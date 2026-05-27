import math


def compute_tfidf(
    ngrams: list[str],
    vocabulary: dict[str, int],
    idf: list[float],
) -> dict[int, float]:
    tf: dict[int, int] = {}
    for g in ngrams:
        idx = vocabulary.get(g)
        if idx is not None:
            tf[idx] = tf.get(idx, 0) + 1

    weighted: dict[int, float] = {
        idx: (1.0 + math.log(count)) * idf[idx] for idx, count in tf.items()
    }

    norm = math.sqrt(sum(w * w for w in weighted.values()))
    if norm > 0:
        weighted = {idx: w / norm for idx, w in weighted.items()}

    return weighted
