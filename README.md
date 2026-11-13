# cpp-ann-index-lab

How close does a dependency-free C++20 approximate nearest neighbour
index come to FAISS-class recall, and at what latency and memory cost?

This repo answers that on the corpus discipline of the earlier
vector-systems-benchmark study: 58,999 embedded text chunks (SQuAD v1.1,
30-word windows), 256-dimensional L2-normalized embeddings
(minishlab/potion-base-8M), 300 held-out queries, exact brute-force
ground truth. Two indexes are implemented from scratch here, in C++20,
standard library only, no external ANN library:

- **IVF**: inverted file index, spherical k-means training, configurable
  nlist and nprobe.
- **Graph**: an HNSW-style layered navigable small-world graph, own
  implementation, simple neighbour selection.

The baseline is brute-force exact search in the same harness. The FAISS
reference numbers from vector-systems-benchmark are used only as a
labelled CROSS-STUDY comparison (different harness, p95 vs p99): FAISS
IndexHNSWFlat (M = 32, efSearch = 40) measured recall@10 0.967 at
p50 0.17 ms on this corpus and query set.

## Headline results

Full tables, verdict rules and caveats: [RESULTS.md](RESULTS.md).
Protocol frozen before any run: [PREREGISTRATION.md](PREREGISTRATION.md).

- **H1 CONFIRMED.** IVF reaches recall@10 0.9487 at its tuned point
  (nlist 512, nprobe 32, chosen by the frozen smallest-nprobe rule) with
  p99 1.28 ms, 8.4x faster than brute force p99 on the same machine.
  The brute-force tail on this shared VM is noisy (see RESULTS.md), and
  the nlist 128 configuration fails the same speed rule: its id-gather
  layout turns random-access bound at high nprobe.
- **H2 CONFIRMED.** At the IVF tuned point's p99 (1.28 ms), the graph
  index reaches recall@10 0.9750 (M 16, ef 128, p99 0.97 ms) and 0.9620
  at p99 0.51 ms (M 32, ef 32): strictly better recall at equal or
  lower latency.
- **H3 CONFIRMED.** Every configuration stays inside the frozen memory
  budget of 1,536 bytes per vector (raw float32 storage is 1,024). The
  heaviest, graph M 32, totals 1,340.8 bytes per vector; IVF needs
  about 1,030 to 1,037.
- Against the CROSS-STUDY FAISS reference (recall@10 0.967), this
  repo's graph index at M 32, ef 32 measures recall@10 0.9620 with
  p50 0.18 ms: within half a point of FAISS recall at a comparable
  operating point, from a standard-library-only implementation.

## Why this matters

Retrieval that runs cheaply on commodity hardware is what makes
semantic search possible in low-resource deployments: a clinic, a
school, a local-language archive, an offline assistant with no GPU, no
cloud account and no database server. An index that builds with a
stock C++ compiler and fits in about 1.3 KB per vector lowers the
floor for who can run retrieval at all.

## Reproduce

Toolchain: a C++20 compiler (g++ 13.3 used here) and CMake 3.20+.
Both were already present on the study machine; nothing was installed.

```sh
# 1. Export the frozen corpus and queries from the source study's
#    cached embeddings (path overridable via ANN_SOURCE_NPZ).
python3 tools/export_embeddings.py

# 2. Build and test.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/ann_tests

# 3. Ground truth, then the full benchmark (about 8 minutes here).
./build/ann_bench --corpus data/corpus.f32 --queries data/queries.f32 \
    --n 58999 --d 256 --nq 300 \
    --gt-out results/ground_truth.json --out results/run_full.json
python3 tools/validate_ground_truth.py

# 4. Regenerate RESULTS.md from the raw run JSON.
python3 tools/make_results.py --corpus-sha <sha256> \
    --queries-sha <sha256> --gt-validation "PASSED ..." 
```

CI runs the build, the unit tests and a smoke benchmark on a small
synthetic slice (the corpus export needs the source study's data, so
CI does not rerun the full study).

## Layout

- `include/ann/` - the library: `vecmath.h` (inner product, top-k),
  `brute_force.h` (exact baseline and ground truth), `ivf.h`,
  `hnsw.h`, `dataset.h` (raw float32 loader).
- `src/ann_bench.cpp` - benchmark harness (frozen protocol).
- `tests/test_ann.cpp` - unit tests: distance functions, top-k
  tie-breaking, brute force vs full-sort ground truth, IVF at full
  probe and the graph index at ef above N both exactly reproduce
  brute force on tiny sets, k-means determinism, loader round trip.
- `tools/` - corpus export, ground-truth validation, RESULTS.md
  generator.
- `results/` - raw per-repetition JSON for both recorded runs and the
  ground-truth file.

## Claims ceiling

Study scope only: one corpus, one machine, one query set, one frozen
protocol. These numbers describe this implementation on this workload.
They are not a claim about production systems, other corpora, or
production experience.

## Licence

Apache-2.0 (see LICENSE). The corpus text is SQuAD v1.1 (CC BY-SA
4.0); embeddings are from minishlab/potion-base-8M (MIT). Corpus
binaries are not committed; the export script reproduces them from
the source study.
