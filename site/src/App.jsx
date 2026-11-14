import data from './data.json';

const REPO = 'https://github.com/JesunAhmadUshno/cpp-ann-index-lab';
const f = (x, d = 3) => Number(x).toFixed(d);
const f4 = (x) => Number(x).toFixed(4);
const ms = (mean, std) => `${f(mean)} +/- ${f(std)}`;

function Chart() {
  // Scatter: recall@10 (y) against p99 latency in ms (x, log scale).
  const W = 780;
  const H = 430;
  const m = { l: 64, r: 24, t: 18, b: 52 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const xMin = Math.log10(0.08);
  const xMax = Math.log10(120);
  const yMin = 0.3;
  const yMax = 1.02;
  const X = (v) => m.l + ((Math.log10(v) - xMin) / (xMax - xMin)) * iw;
  const Y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin)) * ih;
  const tunedKey = `${data.h2.reference.nlist}/${data.h2.reference.nprobe}`;
  const points = [
    ...data.ivf.map((c) => ({
      kind: 'ivf', label: `IVF nlist ${c.nlist}, nprobe ${c.nprobe}`, x: c.p99, y: c.recall,
      tuned: `${c.nlist}/${c.nprobe}` === tunedKey,
    })),
    ...data.hnsw.map((c) => ({
      kind: 'hnsw', label: `Graph M ${c.M}, ef ${c.ef}`, x: c.p99, y: c.recall, tuned: false,
    })),
    { kind: 'brute', label: 'Brute force (exact)', x: data.brute.p99, y: data.brute.recall, tuned: false },
  ];
  const xTicks = [0.1, 1, 10, 100];
  const yTicks = [0.4, 0.6, 0.8, 0.9, 1.0];
  return (
    <figure className="chart-card">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter chart of recall at 10 against p99 latency">
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={X(t)} y1={m.t} x2={X(t)} y2={m.t + ih} className="grid" />
            <text x={X(t)} y={m.t + ih + 22} className="axis" textAnchor="middle">{t}</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.l} y1={Y(t)} x2={m.l + iw} y2={Y(t)} className="grid" />
            <text x={m.l - 10} y={Y(t) + 4} className="axis" textAnchor="end">{f(t, 1)}</text>
          </g>
        ))}
        <line x1={m.l} y1={Y(data.lines.recall)} x2={m.l + iw} y2={Y(data.lines.recall)} className="rule-line" />
        <text x={m.l + iw - 4} y={Y(data.lines.recall) - 6} className="axis rule-label" textAnchor="end">
          H1 recall line {f(data.lines.recall, 2)}
        </text>
        {points.map((p) =>
          p.kind === 'hnsw' ? (
            <rect
              key={p.label}
              x={X(p.x) - 5}
              y={Y(p.y) - 5}
              width="10"
              height="10"
              transform={`rotate(45 ${X(p.x)} ${Y(p.y)})`}
              className="pt-hnsw"
            >
              <title>{`${p.label}: recall@10 ${f4(p.y)}, p99 ${f(p.x)} ms`}</title>
            </rect>
          ) : (
            <circle
              key={p.label}
              cx={X(p.x)}
              cy={Y(p.y)}
              r={p.kind === 'brute' ? 6 : 5}
              className={p.kind === 'brute' ? 'pt-brute' : p.tuned ? 'pt-ivf tuned' : 'pt-ivf'}
            >
              <title>{`${p.label}: recall@10 ${f4(p.y)}, p99 ${f(p.x)} ms`}</title>
            </circle>
          ),
        )}
        <text x={m.l + iw / 2} y={H - 8} className="axis" textAnchor="middle">p99 latency, ms (log scale)</text>
        <text x={16} y={m.t + ih / 2} className="axis" textAnchor="middle" transform={`rotate(-90 16 ${m.t + ih / 2})`}>
          recall@10
        </text>
      </svg>
      <figcaption className="legend">
        <span><i className="dot ivf" /> IVF cells</span>
        <span><i className="dot hnsw" /> Graph (HNSW style) cells</span>
        <span><i className="dot brute" /> Brute force (exact, recall 1.0)</span>
        <span><i className="dot tuned-ring" /> Ring: IVF tuned point (H1 and H2 reference)</span>
      </figcaption>
    </figure>
  );
}

function VerdictCard({ tag, title, verdict, children }) {
  return (
    <article className="card verdict">
      <header>
        <span className="tag">{tag}</span>
        <span className="pill good">{verdict}</span>
      </header>
      <h3>{title}</h3>
      {children}
    </article>
  );
}

export default function App() {
  const d = data;
  const h1Best = d.h1.passing[0];
  const envFacts = [
    ['Corpus', `${d.env.n.toLocaleString('en-US')} chunks, ${d.env.dim} dimensions`],
    ['Embeddings', 'minishlab/potion-base-8M (model2vec), L2-normalized float32'],
    ['Queries', `${d.env.nQueries} held-out chunk embeddings`],
    ['Repetitions', `${d.env.reps} timed per cell after one warm-up pass`],
    ['Machine', d.env.cpu],
    ['Compiler', d.env.compiler],
  ];
  return (
    <div className="page">
      <nav className="top">
        <a className="brand" href={REPO}>cpp-ann-index-lab</a>
        <div className="links">
          <a href={REPO}>GitHub repo</a>
          <a href={`${REPO}/blob/main/RESULTS.md`}>RESULTS.md</a>
          <a href={`${REPO}/blob/main/PREREGISTRATION.md`}>PREREGISTRATION.md</a>
        </div>
      </nav>

      <header className="hero">
        <p className="kicker">Pre-registered study, frozen protocol, run of record</p>
        <h1>C++ ANN Index Lab</h1>
        <p className="question">{d.researchQuestion}</p>
        <p className="sub">
          Two indexes implemented from scratch in C++20 (standard library only), measured against
          brute-force exact search on the full corpus. Every number on this page is generated at
          build time from the repo record: <code>results/run_full.json</code>, <code>RESULTS.md</code> and{' '}
          <code>PREREGISTRATION.md</code>.
        </p>
      </header>

      <section className="facts">
        {envFacts.map(([k, v]) => (
          <div className="fact" key={k}>
            <span className="fact-k">{k}</span>
            <span className="fact-v">{v}</span>
          </div>
        ))}
      </section>

      <h2 className="section-title">Verdicts, by the frozen rules</h2>
      <section className="cards">
        <VerdictCard
          tag="H1"
          title="IVF reaches FAISS-class recall, at least 5x faster than brute force"
          verdict={d.h1.verdict}
        >
          <p>
            Frozen rule: at the tuned nprobe (smallest nprobe reaching recall@10 of at least{' '}
            {f(d.lines.recall, 2)}), the p99 speed ratio against brute force must be at least{' '}
            {f(d.lines.ratio, 1)}x.
          </p>
          <p className="big">
            recall@10 {f4(h1Best.recall)} at p99 {f(h1Best.p99)} ms
          </p>
          <p>
            Tuned point: nlist {h1Best.nlist}, nprobe {h1Best.nprobe}. Brute-force p99 is{' '}
            {f(d.brute.p99)} ms, so the ratio is {f(h1Best.ratio, 2)}x. The other tuned point
            (nlist {d.tuned[0].nlist}, nprobe {d.tuned[0].nprobe}) reaches recall@10{' '}
            {f4(d.tuned[0].recall)} but its ratio is {f(d.tuned[0].ratio, 2)}x and fails the speed
            rule; both are reported.
          </p>
        </VerdictCard>

        <VerdictCard
          tag="H2"
          title="The graph index beats IVF on recall at equal p99 latency"
          verdict={d.h2.verdict}
        >
          <p>
            Reference: the IVF tuned point (nlist {d.h2.reference.nlist}, nprobe{' '}
            {d.h2.reference.nprobe}), recall@10 {f4(d.h2.reference.recall)} at p99{' '}
            {f(d.h2.reference.p99)} ms. A graph point qualifies with p99 at most the reference and
            recall above it.
          </p>
          <p className="big">
            recall@10 {f4(d.h2.best.recall)} at p99 {f(d.h2.best.p99)} ms
          </p>
          <p>
            Best qualifying point: graph M {d.h2.best.M}, ef {d.h2.best.ef}. In total{' '}
            {d.h2.qualifyingCount} of {d.hnsw.length} graph points beat the reference on both axes
            at once.
          </p>
        </VerdictCard>

        <VerdictCard
          tag="H3"
          title="Memory per vector stays within the frozen budget"
          verdict={d.h3.verdict}
        >
          <p>
            Frozen budget: at most {d.lines.budgetBytes.toLocaleString('en-US')} bytes per vector
            in total (corpus vectors attributed to the index plus index structure).
          </p>
          <p className="big">max {f(d.h3.maxTotalBpV, 1)} B/vector</p>
          <p>
            Measured totals run from {f(Math.min(...d.memory.map((m) => m.totalBpV)), 1)} to{' '}
            {f(d.h3.maxTotalBpV, 1)} B/vector across the four index configurations. Every
            configuration is inside the budget.
          </p>
        </VerdictCard>
      </section>

      <h2 className="section-title">Recall against p99 latency, all cells</h2>
      <Chart />
      <p className="note">
        Upper left is better: higher recall, lower tail latency. Two IVF cells at nlist 128 with
        large nprobe fall behind brute force on p99 (random-access bound posting-list scans); the
        study reports that effect rather than engineering it away. Method note: graph neighbour
        selection is the simple variant (keep the M closest candidates), not the FAISS diversity
        heuristic.
      </p>

      <h2 className="section-title">IVF cells</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>nlist</th><th>nprobe</th><th>recall@10</th><th>p50 ms</th><th>p99 ms</th><th>build s (median)</th></tr>
          </thead>
          <tbody>
            {d.ivf.map((c) => (
              <tr key={`${c.nlist}-${c.nprobe}`} className={`${c.nlist}/${c.nprobe}` === `${d.h2.reference.nlist}/${d.h2.reference.nprobe}` ? 'hl' : ''}>
                <td>{c.nlist}</td><td>{c.nprobe}</td><td>{f4(c.recall)}</td>
                <td>{ms(c.p50, c.p50Std)}</td><td>{ms(c.p99, c.p99Std)}</td><td>{f(c.buildS, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">Highlighted row: the tuned point for nlist {d.h2.reference.nlist} (H1 pass, H2 reference).</p>

      <h2 className="section-title">Graph index cells (HNSW style)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>M</th><th>ef</th><th>recall@10</th><th>p50 ms</th><th>p99 ms</th><th>build s (median)</th></tr>
          </thead>
          <tbody>
            {d.hnsw.map((c) => (
              <tr key={`${c.M}-${c.ef}`}>
                <td>{c.M}</td><td>{c.ef}</td><td>{f4(c.recall)}</td>
                <td>{ms(c.p50, c.p50Std)}</td><td>{ms(c.p99, c.p99Std)}</td><td>{f(c.buildS, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="section-title">Memory per vector (H3 evidence)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Index</th><th>Config</th><th>Structure B/vector</th><th>Total B/vector</th><th>Budget</th><th>Within budget</th></tr>
          </thead>
          <tbody>
            {d.memory.map((m) => (
              <tr key={`${m.index}-${m.config}`}>
                <td>{m.index}</td><td>{m.config}</td>
                <td>{f(m.structureBpV, 1)}</td><td>{f(m.totalBpV, 1)}</td>
                <td>{d.lines.budgetBytes.toLocaleString('en-US')}</td>
                <td>{m.totalBpV <= d.lines.budgetBytes ? 'yes' : 'no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Total counts the corpus vectors attributed to the index ({d.env.dim} x 4 ={' '}
        {(d.env.dim * 4).toLocaleString('en-US')} bytes per vector) plus structure bytes per
        vector, counted from container capacities after build.
      </p>

      <h2 className="section-title">CROSS-STUDY: published FAISS reference</h2>
      <section className="card wide">
        <p>
          <span className="pill warn">CROSS-STUDY</span> Comparison only, exactly as labelled in the
          study record. Source: vector-systems-benchmark, its own frozen protocol, Python harness
          calling the FAISS C++ library, 3 repetitions, same corpus and same 300 query vectors.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Source</th><th>Index</th><th>recall@10</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>CROSS-STUDY (vector-systems-benchmark)</td>
                <td>FAISS IndexHNSWFlat, M 32, efSearch 40</td>
                <td>{f(d.crossStudy.faiss.recall)}</td><td>{f(d.crossStudy.faiss.p50, 2)}</td>
                <td>{f(d.crossStudy.faiss.p95, 2)}</td><td>not reported</td>
              </tr>
              <tr>
                <td>This study</td>
                <td>Graph M {d.crossStudy.ours.M}, ef {d.crossStudy.ours.ef} (nearest frozen ef to 40)</td>
                <td>{f4(d.crossStudy.ours.recall)}</td><td>{f(d.crossStudy.ours.p50)}</td>
                <td>not reported</td><td>{f(d.crossStudy.ours.p99)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          Read the comparison with its limits: different harness (Python loop vs C++ loop),
          different headline quantile (p95 vs p99), 3 vs {d.env.reps} repetitions. The recall
          figures share corpus, queries and the ground-truth rule, so the recall gap (
          {f(d.crossStudy.faiss.recall)} vs {f4(d.crossStudy.ours.recall)}) is the most meaningful
          part; latency across the two harnesses is indicative only.
        </p>
        <p>
          Ground truth: this study&apos;s C++ brute force, cross-validated against an independent
          NumPy exact top-10: PASSED, {d.groundTruth.passed} of {d.groundTruth.of} queries,{' '}
          {d.groundTruth.setMismatch} set mismatches, {d.groundTruth.orderMismatch} order
          mismatches.
        </p>
      </section>

      <h2 className="section-title">Honesty notes, carried from the record</h2>
      <section className="card wide">
        <p>
          Tail latency on this shared VM is noisy. The brute-force p99 repetitions in the run of
          record were {d.brute.p99Reps.map((r) => f(r, 2)).join(', ')} ms (mean {f(d.brute.p99)},
          std {f(d.brute.p99Std)}). H1&apos;s speed ratio is computed by the frozen rule from the
          reported means; read the verdict together with this variance. Recall and all p50 figures
          were stable.
        </p>
        <p>
          Two full runs of the identical frozen protocol were recorded (the first ran while a
          validation process shared the VM and was superseded by a clean re-run). Both raw files
          are kept in the repo, and the three verdicts are the same under both runs.
        </p>
      </section>

      <footer>
        <p>
          Claims ceiling: study scope only. One corpus, one machine, one query set, one frozen
          protocol. These numbers describe this implementation on this workload; they are not a
          claim about production systems, other corpora, or production experience.
        </p>
        <p>
          Full record: <a href={`${REPO}/blob/main/RESULTS.md`}>RESULTS.md</a> in{' '}
          <a href={REPO}>JesunAhmadUshno/cpp-ann-index-lab</a>. Protocol frozen in advance in{' '}
          <a href={`${REPO}/blob/main/PREREGISTRATION.md`}>PREREGISTRATION.md</a>.
        </p>
      </footer>
    </div>
  );
}
