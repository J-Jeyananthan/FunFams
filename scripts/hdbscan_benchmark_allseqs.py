#!/usr/bin/env python3
"""HDBSCAN clustering benchmark on all (non-rep) embeddings.

Clusters all sequences directly using HDBSCAN (min_cluster_size=2), writes
.faa files, and evaluates EC purity via calculate_ec_purity.py. Unlike
hdbscan_benchmark.py there is no S90 rep expansion step — every sequence in
the .pt file is clustered and written as-is.

Embeddings are always L2-normalised before clustering. With euclidean metric
this is equivalent to cosine distance on unit vectors; with cosine metric the
L2 normalisation is a no-op (cosine is scale-invariant). Both options produce
identical clusterings — euclidean is faster via BLAS routines.

Example:
    python hdbscan_benchmark_allseqs.py \
        --pt_file      .../all_seqs_embeddings.pt \
        --ec_file      .../extracted_uniprot_ec.csv \
        --output_dir   .../experiments/hdbscan_allseqs
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


def write_faa_files(clusters_of_ids, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.faa"):
        f.unlink()
    for i, members in enumerate(clusters_of_ids):
        with open(out_dir / f"cluster_{i:05d}.faa", "w") as fh:
            for seq_id in members:
                fh.write(f">{seq_id}\nX\n")


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN EC purity benchmark (all seqs, no rep expansion).")
    parser.add_argument("--pt_file",     required=True, help="Path to .pt embeddings file (all sequences)")
    parser.add_argument("--ec_file",     required=True, help="EC annotations CSV (uniprot_id,ec_term)")
    parser.add_argument("--output_dir",  required=True, help="Output dir for .faa files")
    parser.add_argument("--min_samples", type=int, default=None,
                        help="HDBSCAN min_samples; defaults to 2")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Distance metric passed to HDBSCAN (default: euclidean). "
                             "Both are equivalent after L2 normalisation; euclidean is faster.")
    args = parser.parse_args()

    t_start = time.time()

    # Load embeddings
    print("Loading embeddings...")
    seq_ids, embeddings = load_pt_embeddings(Path(args.pt_file))
    print(f"  {len(seq_ids)} sequences, dim={embeddings.shape[1]}")

    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    print(f"Running HDBSCAN (min_cluster_size=2, metric={args.metric})...")
    from sklearn.cluster import HDBSCAN
    ms = args.min_samples if args.min_samples is not None else 2
    labels = HDBSCAN(min_cluster_size=2, min_samples=ms,
                     metric=args.metric).fit_predict(embeddings)

    # noise points (label == -1) each become their own singleton cluster
    cluster_map = defaultdict(list)
    noise_idx = 0
    for seq_id, label in zip(seq_ids, labels):
        if label == -1:
            cluster_map[f"_noise_{noise_idx}"].append(seq_id)
            noise_idx += 1
        else:
            cluster_map[label].append(seq_id)

    clusters = list(cluster_map.values())
    n_clusters = len(clusters)
    n_singletons = sum(1 for c in clusters if len(c) == 1)
    print(f"  {n_clusters} clusters, singleton%={n_singletons / n_clusters * 100:.1f}%")

    print(f"\nWriting .faa files to {args.output_dir} ...")
    write_faa_files(clusters, Path(args.output_dir))

    print("\n--- EC Purity ---")
    ec_script = Path(__file__).resolve().parent / "calculate_ec_purity.py"
    subprocess.run(
        [sys.executable, str(ec_script),
         "--ec-file", args.ec_file,
         "--funfams-dir", args.output_dir],
        check=True,
    )

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
