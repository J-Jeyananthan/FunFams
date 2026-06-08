#!/usr/bin/env python3
"""HDBSCAN clustering benchmark on HUPs S90 rep embeddings.

Clusters S90 reps using HDBSCAN (min_cluster_size=2), expands to full members
via the S90 cluster TSV, writes .faa files, and evaluates EC purity via
calculate_ec_purity.py.

Embeddings are always L2-normalised before clustering. With euclidean metric
this is equivalent to cosine distance on unit vectors; with cosine metric the
L2 normalisation is a no-op (cosine is scale-invariant). Both options produce
identical clusterings — euclidean is faster via BLAS routines.

Example:
    python hc_benchmark.py \
        --pt_file      .../E_starting_contrastive.pt \
        --cluster_tsv  .../HUPs_mmseqs_s90_3.40.50.620_cluster.tsv \
        --ec_file      .../extracted_uniprot_ec.csv \
        --output_dir   .../experiments/hdbscan_contrastive
"""

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load_pt_embeddings(pt_path: Path):
    """Return (list[seq_id], np.ndarray shape (N, D))."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids = [d["label"] for d in data]
    embeddings = np.stack([d["mean_representations"][33].float().numpy() for d in data])
    return ids, embeddings


def load_cluster_tsv(tsv_path: Path):
    """Return {rep_id: [member_ids]} from MMseqs2 cluster TSV."""
    rep_to_members = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep_to_members[parts[0]].append(parts[1])
    return rep_to_members


def write_faa_files(clusters_of_ids, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.faa"):
        f.unlink()
    for i, members in enumerate(clusters_of_ids):
        with open(out_dir / f"cluster_{i:05d}.faa", "w") as fh:
            for seq_id in members:
                fh.write(f">{seq_id}\nX\n")


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN EC purity benchmark.")
    parser.add_argument("--pt_file",     required=True, help="Path to .pt embeddings file")
    parser.add_argument("--cluster_tsv", required=True, help="S90 cluster TSV (rep -> members)")
    parser.add_argument("--ec_file",     required=True, help="EC annotations CSV (uniprot_id,ec_term)")
    parser.add_argument("--output_dir",  required=True, help="Output dir for .faa files")
    parser.add_argument("--min_samples", type=int, default=None,
                        help="HDBSCAN min_samples; defaults to 2")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Distance metric passed to HDBSCAN (default: euclidean). "
                             "Both are equivalent after L2 normalisation; euclidean is faster.")
    args = parser.parse_args()

    t_clustering_start = time.time()

    # Load embeddings
    print("Loading embeddings...")
    rep_ids, embeddings = load_pt_embeddings(Path(args.pt_file))
    print(f"  {len(rep_ids)} reps, dim={embeddings.shape[1]}")

    # L2 normalise — with euclidean metric this gives cosine distance equivalence;
    # with cosine metric it is a no-op (cosine is scale-invariant)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    rep_to_members = load_cluster_tsv(Path(args.cluster_tsv))

    print(f"Running HDBSCAN (min_cluster_size=2, metric={args.metric})...")
    from sklearn.cluster import HDBSCAN
    ms = args.min_samples if args.min_samples is not None else 2
    t_hdbscan_start = time.time()
    labels = HDBSCAN(min_cluster_size=2, min_samples=ms,
                     metric=args.metric).fit_predict(embeddings)
    print(f"  HDBSCAN time: {time.time() - t_hdbscan_start:.1f}s")

    # noise points (label == -1) each become their own singleton cluster
    cluster_map = defaultdict(list)
    noise_idx = 0
    for rep_id, label in zip(rep_ids, labels):
        if label == -1:
            cluster_map[f"_noise_{noise_idx}"].append(rep_id)
            noise_idx += 1
        else:
            cluster_map[label].append(rep_id)

    # Map cluster labels → full member lists via TSV
    expanded = [
        [m for rep in reps for m in rep_to_members.get(rep, [rep])]
        for reps in cluster_map.values()
    ]

    n_clusters = len(expanded)
    n_singletons = sum(1 for c in expanded if len(c) == 1)
    print(f"  {n_clusters} clusters, singleton%={n_singletons / n_clusters * 100:.1f}%")

    print(f"\nWriting .faa files to {args.output_dir} ...")
    write_faa_files(expanded, Path(args.output_dir))

    print(f"\nClustering time: {time.time() - t_clustering_start:.1f}s")

    print("\n--- EC Purity ---")
    ec_script = Path(__file__).resolve().parent / "calculate_ec_purity.py"
    subprocess.run(
        [sys.executable, str(ec_script),
         "--ec-file", args.ec_file,
         "--funfams-dir", args.output_dir],
        check=True,
    )


if __name__ == "__main__":
    main()
