#!/usr/bin/env python3
"""Export the frozen corpus and query vectors to raw float32 binaries.

Reads the vector-systems-benchmark cached artifact (data/embeddings.npz)
and writes:
  data/corpus.f32   array "corpus",  (58999, 256) float32, row-major
  data/queries.f32  array "heldout" rows 0..299, (300, 256) float32

The corpus text is SQuAD v1.1 (CC BY-SA 4.0); the binaries are build
inputs and are not committed to this repo. Reproduce them with:

  python3 tools/export_embeddings.py [--npz PATH] [--out-dir data]

Default --npz resolves the sibling checkout at
~/workspace/gap-closure/vector-systems-benchmark/data/embeddings.npz,
or set ANN_SOURCE_NPZ. Prints SHA-256 of corpus.f32 for the record in
RESULTS.md.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_NPZ = Path.home() / "workspace" / "gap-closure" / \
    "vector-systems-benchmark" / "data" / "embeddings.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.environ.get("ANN_SOURCE_NPZ", str(DEFAULT_NPZ)))
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    z = np.load(args.npz)
    corpus = np.ascontiguousarray(z["corpus"], dtype=np.float32)
    queries = np.ascontiguousarray(z["heldout"][:300], dtype=np.float32)
    assert corpus.shape == (58999, 256), corpus.shape
    assert queries.shape == (300, 256), queries.shape

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.f32"
    queries_path = out / "queries.f32"
    corpus.tofile(corpus_path)
    queries.tofile(queries_path)

    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    qdigest = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    print(f"corpus  {corpus.shape} -> {corpus_path}  sha256={digest}")
    print(f"queries {queries.shape} -> {queries_path}  sha256={qdigest}")


if __name__ == "__main__":
    main()
