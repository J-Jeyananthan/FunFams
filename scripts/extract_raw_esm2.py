"""Export ESM2 embeddings from LMDB to .pt format for use with hdbscan_benchmark.py.

Reads all embeddings from an LMDB database and saves them as a list of dicts with
'label' and 'mean_representations' keys, matching the format expected by hdbscan_benchmark.py.

Usage:
    python scripts/extract_raw_esm2.py \
        --lmdb_path path/to/embeddings.lmdb \
        --output path/to/output.pt \
        --emb_dim 1280
"""

import argparse
import lmdb
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emb_dim", type=int, default=1280)
    args = parser.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    embeddings = []

    with env.begin(write=False) as txn:
        for key, buf in txn.cursor():
            if key == b"__meta__":
                continue
            label = key.decode("utf-8").replace("_", "/", 1)
            vec = np.frombuffer(buf, dtype=np.float16, count=args.emb_dim).copy()
            embeddings.append({
                "label": label,
                "mean_representations": {33: torch.from_numpy(vec)}
            })

    env.close()
    torch.save(embeddings, args.output)
    print(f"Saved {len(embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
