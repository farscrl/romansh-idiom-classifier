/**
 * Computes a sublinear-TF × IDF weighted, L2-normalised feature vector,
 * replicating scikit-learn's TfidfVectorizer with sublinear_tf=True.
 *
 * L2 normalisation is applied independently per vectorizer (char or word),
 * NOT on the concatenated feature vector — this must match the Python pipeline.
 */
export function computeTfidf(
  ngrams: string[],
  vocabulary: Record<string, number>,
  idf: number[],
): Map<number, number> {
  const tf = new Map<number, number>();
  for (const g of ngrams) {
    const idx = vocabulary[g];
    if (idx !== undefined) tf.set(idx, (tf.get(idx) ?? 0) + 1);
  }

  const weighted = new Map<number, number>();
  for (const [idx, count] of tf)
    weighted.set(idx, (1 + Math.log(count)) * idf[idx]);

  let norm = 0;
  for (const w of weighted.values()) norm += w * w;
  norm = Math.sqrt(norm);
  if (norm > 0)
    for (const [idx, w] of weighted) weighted.set(idx, w / norm);

  return weighted;
}