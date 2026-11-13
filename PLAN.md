# PLAN: cpp-ann-index-lab

1. Spec and pre-registration committed first (this commit). The
   pre-registration timestamp is the receipt: no benchmark runs before it.
2. Data route: export script converts the source study's embeddings.npz
   arrays (corpus, heldout rows 0..299) to row-major float32 binaries.
   Data files are not committed (SQuAD is CC BY-SA 4.0; the export script
   plus the source repo reproduce them).
3. Core library (C++20, standard library only):
   - dataset loader for the float32 binaries
   - inner-product and top-k helpers with deterministic tie-breaking
   - brute-force exact search (baseline and ground truth)
   - IVF index (spherical k-means training, posting lists, nprobe scan)
   - HNSW-style graph index (layered graph, ef search)
   - memory accounting per index (capacity-based, post-build)
4. Unit tests: distance functions, brute force vs naive reference on a
   tiny set including tie-breaking, IVF at nprobe = nlist equals brute
   force exactly (recall@10 = 1.0), graph index at ef >= N on a tiny set
   equals brute force (recall@10 = 1.0), k-means determinism under the
   frozen seed, binary loader round trip.
5. Ground-truth cross-validation against an independent NumPy exact
   top-10 over the full query set before benchmarking.
6. Benchmark harness: frozen grids from the pre-registration, warm-up
   pass, 5 timed repetitions per cell, per-repetition JSON under
   results/, plus a synthetic-data smoke mode for CI.
7. RESULTS.md generated from the raw JSON by a script (no hand-typed
   numbers), verdicts by the frozen rules, cross-study FAISS table
   labelled CROSS-STUDY.
8. README with research question, method, headline results, impact.
9. CI: GitHub Actions, CMake configure and build (gcc, C++20, warnings
   on), unit tests, smoke benchmark on a small synthetic slice. CodeQL
   workflow for C++.
10. Claim check: Jev claim_check gate on the RESULTS.md claims; fix and
    re-check until the gate record is saved.
11. Ship: results commit with harness, raw results and RESULTS.md, CI
    green on that commit. Grade recorded outside the public repo.
