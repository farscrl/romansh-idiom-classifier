/**
 * Verify TypeScript classifier against Python test sets.
 *
 * Loads a model JSON and runs predictions on one or more test TSV files,
 * then prints accuracy and macro-F1 — results should match Python step7 exactly.
 *
 * Usage:
 *   pnpm verify ../../models/svm_lite_export.json
 *   pnpm verify ../../models/lr_lite_export.json --tests test_a test_b test_c test_d
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import { RomanshIdiomClassifier } from "../src/index.js";

// -------------------------------------------------------------------------- //
// CLI args                                                                    //
// -------------------------------------------------------------------------- //

const args = process.argv.slice(2);

function arg(flag: string, fallback?: string): string {
  const i = args.indexOf(flag);
  if (i !== -1 && args[i + 1]) return args[i + 1];
  if (fallback !== undefined) return fallback;
  console.error(`Missing required argument: ${flag}`);
  process.exit(1);
}

function argList(flag: string, fallback: string[]): string[] {
  const i = args.indexOf(flag);
  if (i === -1) return fallback;
  const values: string[] = [];
  for (let j = i + 1; j < args.length && !args[j].startsWith("--"); j++)
    values.push(args[j]);
  return values.length ? values : fallback;
}

const modelPath = arg("--model");
const testNames = argList("--tests", ["test_a", "test_b", "test_c", "test_d"]);
const splitsDir = arg("--splits", resolve(import.meta.dirname, "../../../data/03_splits/test"));
const outputPath = args.includes("--output") ? arg("--output") : null;

// -------------------------------------------------------------------------- //
// Metrics                                                                     //
// -------------------------------------------------------------------------- //

function macroF1(trueLabels: string[], predLabels: string[]): { macroF1: number; perClass: Record<string, number> } {
  const classes = [...new Set(trueLabels)].sort();
  const perClass: Record<string, number> = {};

  for (const cls of classes) {
    const tp = trueLabels.filter((t, i) => t === cls && predLabels[i] === cls).length;
    const fp = trueLabels.filter((t, i) => t !== cls && predLabels[i] === cls).length;
    const fn = trueLabels.filter((t, i) => t === cls && predLabels[i] !== cls).length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
    perClass[cls] = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  }

  const macroF1 = Object.values(perClass).reduce((s, v) => s + v, 0) / classes.length;
  return { macroF1, perClass };
}

// -------------------------------------------------------------------------- //
// Main                                                                        //
// -------------------------------------------------------------------------- //

const modelName = modelPath.split("/").pop()?.replace("_export.json", "") ?? modelPath;
const modelData = JSON.parse(readFileSync(resolve(modelPath), "utf-8"));
const classifier = new RomanshIdiomClassifier(modelData);
console.log(`Model loaded: ${modelPath}\n`);

const TEST_SET_LABELS: Record<string, string> = {
  test_a: "Test A — news (FMR)",
  test_b: "Test B — speech transcripts (RTR)",
  test_c: "Test C — schoolbooks (Textbooks)",
  test_d: "Test D — proprietary (out-of-domain)",
};

const jsonTestSets: Record<string, unknown> = {};

for (const testName of testNames) {
  const tsvPath = resolve(splitsDir, `${testName}.tsv`);
  let lines: string[];
  try {
    lines = readFileSync(tsvPath, "utf-8").split("\n").filter(Boolean);
  } catch {
    console.log(`${testName}: file not found — skipping\n`);
    continue;
  }

  const trueLabels: string[] = [];
  const predLabels: string[] = [];

  for (const line of lines) {
    const tab = line.indexOf("\t");
    if (tab === -1) continue;
    const label = line.slice(0, tab);
    const text  = line.slice(tab + 1);
    trueLabels.push(label);
    predLabels.push(classifier.predict(text));
  }

  const correct = trueLabels.filter((t, i) => t === predLabels[i]).length;
  const accuracy = correct / trueLabels.length;
  const { macroF1: mf1, perClass } = macroF1(trueLabels, predLabels);

  console.log(`${"=".repeat(60)}`);
  console.log(`Test set: ${testName}  (${trueLabels.length.toLocaleString()} samples)`);
  console.log(`  accuracy  = ${accuracy.toFixed(4)}`);
  console.log(`  macro-F1  = ${mf1.toFixed(4)}`);
  console.log(`  per-class F1:`);
  for (const [cls, f1] of Object.entries(perClass).sort())
    console.log(`    ${cls.padEnd(16)} ${f1.toFixed(3)}`);
  console.log();

  jsonTestSets[testName] = {
    label:     TEST_SET_LABELS[testName] ?? testName,
    n_samples: trueLabels.length,
    accuracy:  Math.round(accuracy * 1e6) / 1e6,
    macro_f1:  Math.round(mf1 * 1e6) / 1e6,
    per_class: Object.fromEntries(
      Object.entries(perClass).map(([cls, f1]) => [
        cls,
        { f1: Math.round(f1 * 1e6) / 1e6 },
      ])
    ),
  };
}

if (outputPath) {
  const payload = {
    model:     modelName,
    source:    "typescript",
    timestamp: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
    test_sets: jsonTestSets,
  };
  writeFileSync(resolve(outputPath), JSON.stringify(payload, null, 2), "utf-8");
  console.log(`Results saved → ${outputPath}`);
}