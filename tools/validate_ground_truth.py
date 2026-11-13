#!/usr/bin/env python3
"""Cross-validate the C++ ground truth against an independent NumPy
exact top-10, following the source study's rule (inner product on
L2-normalized vectors, ties broken by the smaller row index).

Usage:
  python3 tools/validate_ground_truth.py \
      [--npz PATH] [--gt results/ground_truth.json]

Exit code 0 and a printed agreement line mean the two implementations
agree on the top-10 set of every one of the 300 queries.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_NPZ = Path.home() / "workspace" / "gap-closure" / \
    "vector-systems-benchmark" / "data" / "embeddings.npz"


def numpy_gt(corpus: np.ndarray, queries: np.ndarray, k: int = 10):
    sims = queries @ corpus.T
    out = []
    for row in sims:
        order = sorted(range(corpus.shape[0]),
                       key=lambda j: (-float(row[j]), j))
        out.append(order[:k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.environ.get("ANN_SOURCE_NPZ", str(DEFAULT_NPZ)))
    ap.add_argument("--gt", default="results/ground_truth.json")
    args = ap.parse_args()

    z = np.load(args.npz)
    corpus = np.asarray(z["corpus"], dtype=np.float32)
    queries = np.asarray(z["heldout"][:300], dtype=np.float32)
    expected = numpy_gt(corpus, queries)

    with open(args.gt) as f:
        got = json.load(f)

    assert len(got) == len(expected) == 300
    set_mismatch = 0
    order_mismatch = 0
    for e, g in zip(expected, got):
        if set(e) != set(g):
            set_mismatch += 1
        elif e != g:
            order_mismatch += 1
    print(f"queries=300 set_mismatches={set_mismatch} "
          f"order_only_mismatches={order_mismatch}")
    if set_mismatch:
        raise SystemExit("GROUND TRUTH VALIDATION FAILED")
    print("GROUND TRUTH VALIDATION PASSED")


if __name__ == "__main__":
    main()
