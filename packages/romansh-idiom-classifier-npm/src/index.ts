import { charWbNgrams, wordNgrams } from "./tokenize.js";
import { computeTfidf } from "./tfidf.js";
import { MODEL_DATA } from "./model-data.js";
import type { ModelData } from "./types.js";

export type { ModelData, VectorizerData, SparseRow } from "./types.js";

export class RomanshIdiomClassifier {
  private readonly _charCoef: Map<number, number>[];
  private readonly _wordCoef: Map<number, number>[];

  /** Without arguments, uses the bundled LR-lite model. Pass a parsed model object to use a custom model. */
  constructor(private readonly model: ModelData = MODEL_DATA) {
    this._charCoef = model.char_coef.map(({ idx, val }) => {
      const m = new Map<number, number>();
      for (let i = 0; i < idx.length; i++) m.set(idx[i], val[i]);
      return m;
    });
    this._wordCoef = model.word_coef.map(({ idx, val }) => {
      const m = new Map<number, number>();
      for (let i = 0; i < idx.length; i++) m.set(idx[i], val[i]);
      return m;
    });
  }

  /**
   * Predict the Romansh idiom of the given text.
   * Returns one of: rm-sursilv, rm-sutsilv, rm-surmiran, rm-puter, rm-vallader, rm-rumgr, de, fr, it, en
   */
  predict(text: string): string {
    const scores = this.score(text);
    return this.model.classes.reduce((best, c) => (scores[c] > scores[best] ? c : best));
  }

  /**
   * Return a raw decision score per class (higher = more confident).
   * Scores are unbounded real numbers — positive means evidence for that idiom,
   * negative means evidence against. Use the gap between scores to judge confidence.
   */
  score(text: string): Record<string, number> {
    const { classes, char, word, intercept } = this.model;

    const charF = computeTfidf(
      charWbNgrams(text, char.ngram_range[0], char.ngram_range[1]),
      char.vocabulary,
      char.idf,
    );
    const wordF = computeTfidf(
      wordNgrams(text, word.ngram_range[0], word.ngram_range[1]),
      word.vocabulary,
      word.idf,
    );

    const result: Record<string, number> = {};
    for (let c = 0; c < classes.length; c++) {
      let s = intercept[c];
      const charCoef = this._charCoef[c];
      charF.forEach((w, idx) => {
        const val = charCoef.get(idx);
        if (val !== undefined) s += w * val;
      });
      const wordCoef = this._wordCoef[c];
      wordF.forEach((w, idx) => {
        const val = wordCoef.get(idx);
        if (val !== undefined) s += w * val;
      });
      result[classes[c]] = s;
    }
    return result;
  }

  /**
   * Return softmax-normalised scores per class (values between 0 and 1, summing to 1).
   * Useful for displaying confidence bars. Note: these are NOT calibrated probabilities —
   * they are a display-friendly transformation of the raw scores.
   */
  softScores(text: string): Record<string, number> {
    const raw = this.score(text);
    const values = Object.values(raw);
    const max = Math.max(...values);
    const exps = values.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const result: Record<string, number> = {};
    Object.keys(raw).forEach((k, i) => result[k] = exps[i] / sum);
    return result;
  }
}