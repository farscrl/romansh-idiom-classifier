#!/usr/bin/env node
/**
 * Run franc on every text in a TSV file and write predictions as JSON to stdout.
 *
 * Input:  TSV with format  <label>\t<text>  (one sample per line)
 * Output: {"labels": [...], "predictions": [...]}
 *
 * Usage:
 *   node franc_predict.js <tsv_file>                  # Romansh only
 *   node franc_predict.js <tsv_file> --multilingual   # Romansh + de/fr/it/en
 */
'use strict';

const franc = require('./all');
const fs = require('fs');

const FRANC_TO_PROJECT = {
    roh_rumgr:    'rm-rumgr',
    roh_puter:    'rm-puter',
    roh_surmiran: 'rm-surmiran',
    roh_sursilv:  'rm-sursilv',
    roh_sutsilv:  'rm-sutsilv',
    roh_vallader: 'rm-vallader',
    deu: 'de',
    fra: 'fr',
    ita: 'it',
    eng: 'en',
};

const ROMANSH_CODES    = ['roh_rumgr', 'roh_puter', 'roh_surmiran', 'roh_sursilv', 'roh_sutsilv', 'roh_vallader'];
const MULTILINGUAL_CODES = [...ROMANSH_CODES, 'deu', 'fra', 'ita', 'eng'];

const tsvPath = process.argv[2];
const multilingual = process.argv.includes('--multilingual');

if (!tsvPath) {
    process.stderr.write('Usage: node franc_predict.js <tsv_file> [--multilingual]\n');
    process.exit(1);
}

const whitelist = multilingual ? MULTILINGUAL_CODES : ROMANSH_CODES;

function predict(text) {
    const results = franc.all(text, { whitelist });
    const best = results[0][0];
    return best === 'und' ? null : (FRANC_TO_PROJECT[best] ?? null);
}

const rawLines = fs.readFileSync(tsvPath, 'utf-8').split('\n');
const labels = [];
const predictions = [];

for (const line of rawLines) {
    const trimmed = line.trimEnd();
    if (!trimmed) continue;
    const tab = trimmed.indexOf('\t');
    if (tab === -1) continue;
    labels.push(trimmed.slice(0, tab));
    predictions.push(predict(trimmed.slice(tab + 1)));
}

process.stdout.write(JSON.stringify({ labels, predictions }) + '\n');