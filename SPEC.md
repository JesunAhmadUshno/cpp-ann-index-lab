# SPEC: cpp-ann-index-lab

## Research question

On the same corpus discipline as the earlier vector-systems-benchmark study
(58,999 embedded text chunks, 256 dimensions), how close does a
dependency-free modern C++ (C++20) approximate nearest neighbour (ANN) index
come to FAISS-class recall, and at what latency and memory cost?

## Scope

- Two ANN indexes, implemented from scratch in this repo, standard library
  only, no external ANN library:
  - (a) an inverted file (IVF) index with configurable nlist and nprobe,
    trained with spherical k-means.
  - (b) a graph index in the HNSW style (layered navigable small world
    graph), own implementation.
- One exact baseline: brute-force inner-product search in C++, which also
  produces the ground truth every recall number is measured against.
- One external reference: the published FAISS numbers from
  vector-systems-benchmark, used only as a labelled CROSS-STUDY comparison.
  Different harness, different language runtime, different latency quantile
  (p95 there, p99 here). Never presented as a head-to-head run.

## Claims ceiling

Study scope only. All numbers are measurements on one corpus, one machine,
one query set, under the frozen protocol in PREREGISTRATION.md. Nothing in
this repo is a claim about production systems or production experience.

## Societal impact (named)

Retrieval that runs cheaply on commodity hardware matters for low-resource
deployments: small clinics, schools, local-language archives and offline
assistants that cannot assume a GPU, a cloud account, or a heavyweight
database server. A dependency-free index that builds with a stock compiler
lowers the floor for who can run semantic search at all.

## Engineering pipeline

spec, plan, build, test, review, ship. Hypotheses and the full measurement
protocol are frozen in PREREGISTRATION.md, committed before any benchmark
run. Verdicts publish either way: CONFIRMED, REFUTED, or INCONCLUSIVE by
the frozen rules.
