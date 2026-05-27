// Matches sklearn's default token_pattern=r"(?u)\b\w\w+\b":
// Unicode letters/digits/underscore, minimum 2 characters.
const TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;

/**
 * Replicates scikit-learn's char_wb analyzer: tokenises using the same
 * Unicode-aware word pattern as sklearn, pads each token with spaces, then
 * extracts character n-grams within each token (never crossing word boundaries).
 */
export function charWbNgrams(text: string, minN: number, maxN: number): string[] {
  const ngrams: string[] = [];
  for (const token of text.toLowerCase().match(TOKEN_RE) ?? []) {
    const padded = " " + token + " ";
    for (let n = minN; n <= maxN; n++)
      for (let i = 0; i <= padded.length - n; i++)
        ngrams.push(padded.slice(i, i + n));
  }
  return ngrams;
}

/**
 * Replicates scikit-learn's word analyzer: lowercases, extracts tokens with
 * the same Unicode-aware 2+-char pattern as sklearn, then builds word n-grams.
 */
export function wordNgrams(text: string, minN: number, maxN: number): string[] {
  const tokens = text.toLowerCase().match(TOKEN_RE) ?? [];
  const ngrams: string[] = [];
  for (let n = minN; n <= maxN; n++)
    for (let i = 0; i <= tokens.length - n; i++)
      ngrams.push(tokens.slice(i, i + n).join(" "));
  return ngrams;
}