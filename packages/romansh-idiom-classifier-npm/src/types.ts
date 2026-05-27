export interface SparseRow {
  idx: number[];
  val: number[];
}

export interface VectorizerData {
  ngram_range: [number, number];
  vocabulary: Record<string, number>;
  idf: number[];
}

export interface ModelData {
  classes: string[];
  char: VectorizerData;
  word: VectorizerData;
  char_coef: SparseRow[];
  word_coef: SparseRow[];
  intercept: number[];
}