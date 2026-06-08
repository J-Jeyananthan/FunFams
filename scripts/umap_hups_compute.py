#!/usr/bin/env python3
"""Compute UMAP coords for HUPs S90 rep embeddings and save to .npz.

Run this on the HPC where the LMDB lives (~3,700 S90 cluster reps).
The output .npz is tiny and can be copied to a laptop to plot with
umap_hups_plot.py, so you can iterate on aesthetics and EC colour level
without re-running UMAP.

Usage:
    python scripts/umap_hups_compute.py \
        --lmdb_path   .../HUPs_mmseqs_s90_3.40.50.620_rep_seq.lmdb \
        --ec_file     .../extracted_uniprot_ec.csv \
        --output      umap_coords.npz \
        [--emb_dim 1024] \
        [--n_neighbors 15] \
        [--min_dist 0.1] \
        [--seed 42]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
import umap


def load_ec_annotations(ec_file: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = defaultdict(set)
    with open(ec_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            uid, ec = parts[0].strip(), parts[1].strip()
            annotations[uid].add(ec)
    return dict(annotations)


def load_pt_embeddings(pt_path: Path):
    """Return (ids, embeddings) from a .pt file (list of dicts with label + mean_representations)."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids = [d["label"].split("/")[0] for d in data]
    embeddings = np.stack([d["mean_representations"][33].float().numpy() for d in data])
    return ids, embeddings


def load_lmdb_embeddings(lmdb_path: Path, emb_dim: int):
    """Return (ids, embeddings) where ids are UniProt accessions."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    ids = []
    vecs = []
    with env.begin(write=False) as txn:
        for key, buf in txn.cursor():
            if key == b"__meta__":
                continue
            lmdb_key = key.decode("utf-8")
            uniprot_id = lmdb_key.split("_")[0]
            vec = np.frombuffer(buf, dtype=np.float16, count=emb_dim).copy().astype(np.float32)
            ids.append(uniprot_id)
            vecs.append(vec)
    env.close()
    return ids, np.stack(vecs)


def main():
    parser = argparse.ArgumentParser(description="Compute UMAP coords for HUPs S90 rep embeddings.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lmdb_path", help="Path to LMDB directory")
    source.add_argument("--pt_file",   help="Path to .pt embeddings file")
    parser.add_argument("--ec_file",     required=True, help="EC annotations CSV (uniprot_id,ec_term)")
    parser.add_argument("--output",      required=True, help="Output .npz path")
    parser.add_argument("--emb_dim",     type=int, default=1024)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist",    type=float, default=0.1)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.pt_file:
        print(f"Loading embeddings from {args.pt_file} ...")
        ids, embeddings = load_pt_embeddings(Path(args.pt_file))
    else:
        print(f"Loading embeddings from {args.lmdb_path} ...")
        ids, embeddings = load_lmdb_embeddings(Path(args.lmdb_path), args.emb_dim)
    print(f"  {len(ids):,} embeddings, dim={embeddings.shape[1]}")

    print(f"Loading EC annotations from {args.ec_file} ...")
    ec_annotations = load_ec_annotations(Path(args.ec_file))
    print(f"  {len(ec_annotations):,} UniProt IDs with EC annotations")

    # Store raw EC terms (pipe-separated) so the plot script can assign labels at any level
    ec_terms = [
        "|".join(sorted(ec_annotations[uid])) if uid in ec_annotations else ""
        for uid in ids
    ]
    n_with_ec = sum(1 for t in ec_terms if t)
    print(f"  {n_with_ec:,} / {len(ids):,} reps have EC annotations")

    # L2 normalise (cosine equivalence with euclidean metric)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.where(norms == 0, 1.0, norms)

    print(f"Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}) ...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.seed,
        low_memory=True,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"  UMAP done, shape={coords.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coords=coords.astype(np.float32),
        ec_terms=np.array(ec_terms),
        ids=np.array(ids),
    )
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Copy to laptop and run: python scripts/umap_hups_plot.py --npz {out_path.name} --color_level 1")


if __name__ == "__main__":
    main()
