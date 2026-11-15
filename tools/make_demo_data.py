#!/usr/bin/env python3
"""Generate a small synthetic corpus and query set for ann_query demos.

Pure standard library, deterministic (fixed seed). Vectors are drawn
from well-separated clusters on the unit sphere, so approximate search
has real structure to find and the top-10 output is visibly sensible.
Queries are noisy copies of random corpus vectors.

Output format matches tools/export_embeddings.py: headerless
row-major float32, little-endian.

Usage:
    python3 tools/make_demo_data.py --out-dir demo
    # writes demo/corpus.f32 (n x d) and demo/queries.f32 (nq x d)
"""

import argparse
import array
import math
import os
import random


def unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="demo")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261115)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    centers = [unit([rng.gauss(0.0, 1.0) for _ in range(args.d)])
               for _ in range(args.clusters)]

    corpus = []
    for i in range(args.n):
        center = centers[i % args.clusters]
        corpus.append(unit([c + rng.gauss(0.0, 0.15) for c in center]))

    queries = []
    for _ in range(args.nq):
        src = corpus[rng.randrange(args.n)]
        queries.append(unit([x + rng.gauss(0.0, 0.05) for x in src]))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("corpus", corpus), ("queries", queries)):
        flat = array.array("f", (x for row in rows for x in row))
        path = os.path.join(args.out_dir, name + ".f32")
        with open(path, "wb") as fh:
            flat.tofile(fh)
        print(f"wrote {path}: {len(rows)} x {args.d}")


if __name__ == "__main__":
    main()
