# Pre-registration: cpp-ann-index-lab

Frozen before any benchmark run. The git commit that introduces this file
is the receipt: it precedes all benchmark code execution in this repo.
Any change to this file after results exist must be recorded as an
amendment with a date and a reason, never a silent edit.

## Corpus (exact)

- Source text: SQuAD v1.1 train split (licence CC BY-SA 4.0), paragraphs
  chunked into 30-word windows by the vector-systems-benchmark pipeline.
- Indexed set: 58,999 chunks. This study reuses the source study's cached
  embedding artifact verbatim: `data/embeddings.npz` of
  JesunAhmadUshno/vector-systems-benchmark, array key `corpus`,
  shape (58999, 256), dtype float32.
- Embeddings: minishlab/potion-base-8M via model2vec 0.10.0 (model licence
  MIT), 256 dimensions, L2-normalized at creation in the source study, so
  cosine similarity equals inner product. This study does not re-embed.
- The exported binary corpus file used by the harness (row-major float32)
  will have its SHA-256 recorded in RESULTS.md.

## Query set (exact)

- 300 query vectors: rows 0 to 299 of the `heldout` array of the same
  artifact (embeddings of held-out SQuAD chunks, same model, same
  normalization). These are the same 300 query vectors the source study
  used for its unfiltered recall cells.

## Ground truth

- Exact brute-force top-10 by inner product over all 58,999 corpus
  vectors, computed by this repo's C++ brute-force implementation.
- Ties broken by the smaller row index, deterministically (the source
  study's rule).
- Before benchmarking, the C++ ground truth is cross-validated against an
  independent NumPy exact top-10 computation over the full 300-query set;
  the comparison outcome is recorded in RESULTS.md. If the two disagree,
  benchmarking stops until the disagreement is resolved.

## Similarity

Inner product on the L2-normalized vectors (cosine equivalent) everywhere:
brute force, IVF, graph index, ground truth.

## Indexes and frozen grids

All index code is written in this repo. No external ANN library.

- Brute force (baseline and ground truth): full scan, exact.
- IVF: spherical k-means training (assignment by maximum inner product,
  centroids renormalized each iteration), 20 Lloyd iterations, training
  sample = the first 10,000 corpus vectors, k-means RNG seed 20261113.
  - nlist in {128, 512}.
  - nprobe in {1, 2, 4, 8, 16, 32, 64}, capped at nlist.
- Graph index (HNSW style, own implementation): level multiplier
  1 / ln(M), level RNG seed 20261114, efConstruction = 200.
  - M in {16, 32}. Layer-0 link cap = 2M, upper-layer link cap = M.
  - ef (search) in {16, 32, 64, 128, 256}.

## Hypotheses and verdict rules

### H1 (IVF speed at FAISS-class recall)

IVF reaches recall@10 of at least 0.90 at tuned nprobe, with p99 latency
at least 5x faster than brute force at the same corpus scale.

- Tuning rule (frozen, not cherry-picking): for each nlist, the tuned
  nprobe is the SMALLEST nprobe in the frozen ladder whose mean recall@10
  (across the 5 repetitions) is at least 0.90. If no nprobe reaches 0.90,
  that nlist has no tuned point.
- Speed rule: at a tuned point, the speed ratio is brute-force mean p99
  divided by IVF mean p99, same machine, same repetitions.
- CONFIRMED: at least one nlist has a tuned point with speed ratio
  at least 5.0.
- REFUTED: no nlist reaches recall@10 0.90 at any nprobe, or every tuned
  point has speed ratio below 5.0.
- INCONCLUSIVE: only if a needed cell cannot run (for example memory),
  with the failure recorded.

### H2 (graph beats IVF at equal latency)

The graph index beats IVF on recall@10 at equal p99 latency.

- Reference point: the IVF tuned point selected by the H1 rule, choosing
  the nlist whose tuned point has the lowest p99 latency. If the H1 rule
  selects no tuned point, the reference is the IVF grid point with the
  highest mean recall@10, and that substitution is stated in RESULTS.md.
- CONFIRMED: at least one graph operating point (frozen grid) has mean
  p99 latency at most the reference point's mean p99, and mean recall@10
  strictly greater than the reference point's mean recall@10.
- REFUTED: every graph point with p99 at most the reference p99 has
  recall@10 at most the reference recall@10.
- INCONCLUSIVE: the latency ranges do not overlap at all (every graph
  point is slower than the reference point), so no equal-latency
  comparison exists.

### H3 (memory budget)

Memory per vector stays within a pre-set budget.

- Raw storage of one vector: 256 dimensions x 4 bytes (float32) = 1,024
  bytes per vector.
- Budget: total standalone index memory per vector at most 1,536 bytes,
  that is the raw vector bytes times a stated overhead factor of 1.5.
- Total standalone index memory = the corpus vector array attributed to
  the index (N x 1,024 bytes, as any deployed index must hold its
  vectors) plus every byte of index structure the implementation owns
  (posting lists, centroids, graph links, offsets, ids), measured by the
  implementation's own accounting of container capacities after build.
- Judged per index configuration (IVF per nlist, graph per M) at every
  benchmarked configuration: CONFIRMED if all configurations are at or
  under 1,536 bytes per vector, REFUTED if any configuration exceeds it.
  Per-configuration numbers are reported either way.

## Measurement protocol

- Machine: AMD EPYC 9D25 126-Core Processor, 4 vCPUs visible to this VM,
  about 7.7 GiB RAM, Ubuntu 24.04, g++ 13.3.0, CMake Release build,
  flags -O3, C++20, no -march=native (portability; identical flags for
  every arm, so ratios are fair).
- Single-threaded search. Queries processed one at a time in fixed order
  (query 0 to 299) in every repetition.
- One full untimed warm-up pass over the query set per index
  configuration before timed repetitions.
- 5 timed repetitions per search cell. Per-query latency by
  std::chrono::steady_clock. Percentiles per repetition by the
  nearest-rank method over that repetition's 300 latencies.
- Reporting: mean plus standard deviation across the 5 repetitions for
  recall@10, p50 latency and p99 latency. Recall of a fixed index is
  deterministic, so its standard deviation is expected to be 0; it is
  reported anyway.
- Build time: IVF and brute force, median of 5 builds. Graph index,
  median of 3 builds per M. The count actually used is stated in
  RESULTS.md.
- Raw per-repetition outputs are saved under results/ as JSON.

## Cross-study FAISS reference (labelled CROSS-STUDY wherever used)

From vector-systems-benchmark RESULTS.md (its own frozen protocol):
FAISS IndexHNSWFlat, M = 32, efConstruction = 200, efSearch = 40,
unfiltered cell, same corpus and same 300 query vectors: recall@10
0.967, p50 0.17 ms, p95 0.47 ms, 300 queries x 3 repetitions, Python
harness calling the FAISS C++ library. Differences from this study:
harness language and timing loop, latency quantile (p95 vs p99),
repetition count. It is a reference point, not a head-to-head run, and
the earlier study ran on the same class of VM (AMD EPYC 9D25).

## Verdict publication

Verdicts publish either way, including REFUTED and INCONCLUSIVE, with
the numbers that decided them. No operating point outside the frozen
grids may be reported as a result; exploratory observations, if any,
are labelled exploratory and carry no verdict weight.
