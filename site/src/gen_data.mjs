// Generates src/data.json from the study record at build time:
// results/run_full.json (measured cells), RESULTS.md (environment table,
// cross-study table, ground-truth validation), PREREGISTRATION.md (frozen
// lines) and SPEC.md (research question). Verdicts are recomputed here from
// the frozen rules, never typed. The build fails if the record does not
// parse, so the site cannot drift from the committed numbers.
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const srcDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(srcDir, '..', '..');

const run = JSON.parse(readFileSync(path.join(repoRoot, 'results', 'run_full.json'), 'utf8'));
const resultsMd = readFileSync(path.join(repoRoot, 'RESULTS.md'), 'utf8');
const prereg = readFileSync(path.join(repoRoot, 'PREREGISTRATION.md'), 'utf8');
const spec = readFileSync(path.join(repoRoot, 'SPEC.md'), 'utf8');

function need(re, text, label) {
  const m = text.match(re);
  if (!m) throw new Error(`gen_data: could not parse ${label}`);
  return m;
}
const num = (s) => Number(String(s).replace(/,/g, ''));

// Frozen lines from the preregistration.
const recallLine = num(need(/recall@10 of at least (\d+(?:\.\d+)?)/, prereg, 'H1 recall line')[1]);
const ratioLine = num(need(/speed ratio\s+at least (\d+(?:\.\d+)?)/, prereg, 'H1 ratio line')[1]);
const budgetBytes = num(need(/at most ([\d,]+) bytes/, prereg, 'H3 budget')[1]);
for (const [label, v] of [['recallLine', recallLine], ['ratioLine', ratioLine], ['budgetBytes', budgetBytes]]) {
  if (!Number.isFinite(v)) throw new Error(`gen_data: frozen line ${label} did not parse to a number`);
}

// Research question from the spec.
const researchQuestion = need(/## Research question\s+([\s\S]*?)\n\n/, spec, 'research question')[1]
  .replace(/\s+/g, ' ')
  .trim();

// Environment rows from RESULTS.md.
const envRow = (label) => need(new RegExp(`^\\| ${label} \\| (.*) \\|$`, 'm'), resultsMd, `env row ${label}`)[1];
const env = {
  n: run.env.n,
  dim: run.env.dim,
  nQueries: run.env.n_queries,
  reps: run.env.reps,
  cpu: envRow('CPU'),
  ram: envRow('RAM'),
  os: envRow('OS'),
  compiler: envRow('Compiler'),
  corpus: envRow('Corpus'),
  embeddings: envRow('Embeddings'),
};

// Ground-truth validation line from RESULTS.md.
const gt = need(
  /cross-validated against an independent NumPy exact top-10: PASSED, (\d+) of (\d+) queries, (\d+) set mismatches, (\d+) order mismatches/,
  resultsMd,
  'ground truth validation',
);
const groundTruth = { passed: num(gt[1]), of: num(gt[2]), setMismatch: num(gt[3]), orderMismatch: num(gt[4]) };

// Cross-study FAISS row from RESULTS.md (CROSS-STUDY section).
const fx = need(
  /\| FAISS IndexHNSWFlat M32 efSearch 40 \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| not reported \|/,
  resultsMd,
  'FAISS cross-study row',
);
const faiss = { recall: num(fx[1]), p50: num(fx[2]), p95: num(fx[3]) };

// Cells from the run of record.
const trim = (c) => ({
  params: c.params,
  buildS: c.build_time_s_median,
  structureBytes: c.structure_bytes,
  totalBpV: c.total_bytes_per_vector,
  recall: c.recall_mean,
  recallStd: c.recall_std,
  p50: c.p50_ms_mean,
  p50Std: c.p50_ms_std,
  p99: c.p99_ms_mean,
  p99Std: c.p99_ms_std,
});
const bruteCell = run.cells.find((c) => c.index === 'brute_force');
const ivfCells = run.cells.filter((c) => c.index === 'ivf').map(trim);
const hnswCells = run.cells.filter((c) => c.index === 'hnsw').map(trim);
if (ivfCells.length !== 14 || hnswCells.length !== 10 || !bruteCell) {
  throw new Error(`gen_data: unexpected cell counts ivf=${ivfCells.length} hnsw=${hnswCells.length}`);
}
const brute = {
  recall: bruteCell.recall_mean,
  p50: bruteCell.p50_ms_mean,
  p50Std: bruteCell.p50_ms_std,
  p99: bruteCell.p99_ms_mean,
  p99Std: bruteCell.p99_ms_std,
  p99Reps: bruteCell.reps.map((r) => r.p99_ms),
};

// Frozen tuning rule: per nlist, the smallest nprobe reaching the recall line.
const tuned = [];
for (const nlist of [...new Set(ivfCells.map((c) => c.params.nlist))].sort((a, b) => a - b)) {
  const ladder = ivfCells
    .filter((c) => c.params.nlist === nlist)
    .sort((a, b) => a.params.nprobe - b.params.nprobe);
  const hit = ladder.find((c) => c.recall >= recallLine);
  if (hit) {
    tuned.push({
      nlist,
      nprobe: hit.params.nprobe,
      recall: hit.recall,
      p50: hit.p50,
      p99: hit.p99,
      ratio: brute.p99 / hit.p99,
    });
  }
}

// H1: at least one tuned point with speed ratio at least the frozen line.
const h1Passing = tuned.filter((t) => t.ratio >= ratioLine);
const h1 = {
  verdict: h1Passing.length > 0 ? 'CONFIRMED' : 'REFUTED',
  passing: h1Passing,
};

// H2 reference: the passing tuned point (nlist 512 in the run of record).
const reference = (h1Passing.length ? h1Passing : tuned).reduce((a, b) => (b.nlist > a.nlist ? b : a));
const h2Qualifying = hnswCells
  .filter((c) => c.p99 <= reference.p99 && c.recall > reference.recall)
  .map((c) => ({ M: c.params.M, ef: c.params.ef, recall: c.recall, p50: c.p50, p99: c.p99 }));
const h2Best = h2Qualifying.reduce((a, b) => (b.recall > a.recall ? b : a), h2Qualifying[0] || null);
const h2 = {
  verdict: h2Qualifying.length > 0 ? 'CONFIRMED' : 'REFUTED',
  reference: { nlist: reference.nlist, nprobe: reference.nprobe, recall: reference.recall, p99: reference.p99 },
  qualifyingCount: h2Qualifying.length,
  qualifying: h2Qualifying,
  best: h2Best,
};

// H3: memory configurations (one row per distinct index configuration).
const memSeen = new Map();
for (const c of run.cells) {
  if (c.index === 'ivf') {
    const key = `ivf-${c.params.nlist}`;
    if (!memSeen.has(key))
      memSeen.set(key, { index: 'IVF', config: `nlist ${c.params.nlist}`, structureBytes: c.structure_bytes, structureBpV: c.structure_bytes / run.env.n, totalBpV: c.total_bytes_per_vector });
  }
  if (c.index === 'hnsw') {
    const key = `hnsw-${c.params.M}`;
    if (!memSeen.has(key))
      memSeen.set(key, { index: 'Graph (HNSW style)', config: `M ${c.params.M}`, structureBytes: c.structure_bytes, structureBpV: c.structure_bytes / run.env.n, totalBpV: c.total_bytes_per_vector });
  }
}
const memory = [...memSeen.values()];
const h3 = {
  verdict: memory.every((m) => m.totalBpV <= budgetBytes) ? 'CONFIRMED' : 'REFUTED',
  maxTotalBpV: Math.max(...memory.map((m) => m.totalBpV)),
};

// Cross-study counterpart from this study: graph M32, ef 32 (nearest frozen ef to 40).
const csCell = hnswCells.find((c) => c.params.M === 32 && c.params.ef === 32);
if (!csCell) throw new Error('gen_data: cross-study graph cell M32 ef32 not found');

const data = {
  researchQuestion,
  env,
  lines: { recall: recallLine, ratio: ratioLine, budgetBytes },
  brute,
  ivf: ivfCells.map((c) => ({
    nlist: c.params.nlist, nprobe: c.params.nprobe, recall: c.recall, recallStd: c.recallStd,
    p50: c.p50, p50Std: c.p50Std, p99: c.p99, p99Std: c.p99Std, buildS: c.buildS,
  })),
  hnsw: hnswCells.map((c) => ({
    M: c.params.M, ef: c.params.ef, recall: c.recall, recallStd: c.recallStd,
    p50: c.p50, p50Std: c.p50Std, p99: c.p99, p99Std: c.p99Std, buildS: c.buildS,
  })),
  tuned,
  h1,
  h2,
  memory,
  h3,
  crossStudy: {
    faiss,
    ours: { M: 32, ef: 32, recall: csCell.recall, p50: csCell.p50, p99: csCell.p99 },
  },
  groundTruth,
};

writeFileSync(path.join(srcDir, 'data.json'), JSON.stringify(data, null, 2) + '\n');
console.log(
  `gen_data: H1 ${h1.verdict} (tuned ratios ${tuned.map((t) => t.ratio.toFixed(2) + 'x').join(', ')}), ` +
    `H2 ${h2.verdict} (${h2.qualifyingCount} qualifying points), H3 ${h3.verdict} (max ${h3.maxTotalBpV.toFixed(1)} B/vector of ${budgetBytes})`,
);
