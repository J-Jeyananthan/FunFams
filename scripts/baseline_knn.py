"""Baseline k-NN evaluation on raw ProstT5 embeddings (no projection head)."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from callbacks import compute_metrics, l2_normalize_np
from data import FunfamEmbeddingDataset
from utils import load_funfam_labels, load_h5_keys_from_fasta, resolve_fasta_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def gpu_search(train_vecs: np.ndarray, queries: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    """Chunked matmul k-NN search on GPU (or CPU). Returns indices shape (N, k)."""
    n_train = train_vecs.shape[0]
    db_t = torch.from_numpy(train_vecs).to(device)

    if device.type == "cuda":
        free_mem = torch.cuda.mem_get_info(device)[0]
        target_bytes = int(free_mem * 0.50)
    else:
        target_bytes = 1024 ** 3
    chunk_q = max(1, min(len(queries), int(target_bytes // (n_train * 4))))
    logger.info(f"  query chunk size: {chunk_q} (device: {device})")

    all_indices = torch.empty((len(queries), k), dtype=torch.long)
    for q0 in range(0, len(queries), chunk_q):
        q1 = min(q0 + chunk_q, len(queries))
        q_chunk = torch.from_numpy(queries[q0:q1]).to(device)
        scores = q_chunk @ db_t.T  # (chunk, N)
        all_indices[q0:q1] = scores.topk(k, dim=1).indices.cpu()
        del q_chunk, scores

    return all_indices.numpy()

# Default paths (match configs/train.yaml)
DEFAULTS = {
    "lmdb_path": "/SAN/orengolab/functional-families/janu/contrasted-ff/funfams-4.3-c123.lmdb",
    "label_file": "/SAN/orengolab/functional-families/janu/data/duplicates_removed-funfams-4.3-c123-mapping.txt",
    "train_fasta": "/SAN/orengolab/functional-families/janu/contrasted-ff/train_small_funfams_reincluded.fasta",
    "test_fasta": "/SAN/orengolab/functional-families/janu/contrasted-ff/seq_sim_splits/test_sets/combined_set_s50_to_s90/s50_to_s90_combined.fasta",
}


def collect_embeddings(dataset: FunfamEmbeddingDataset, batch_size: int):
    """Stream dataset into numpy arrays of embeddings and labels."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_emb, all_lab = [], []
    seen = 0
    for emb, lab in dl:
        all_emb.append(emb.numpy().astype(np.float32, copy=False))
        all_lab.append(lab.numpy().astype(np.int64, copy=False))
        seen += lab.shape[0]
        if seen % (batch_size * 500) == 0:
            logger.info(f"  loaded {seen:,} embeddings…")
    return np.concatenate(all_emb), np.concatenate(all_lab)


def main():
    parser = argparse.ArgumentParser(description="Baseline k-NN on raw ProstT5 embeddings")
    parser.add_argument("--lmdb_path", type=str, default=DEFAULTS["lmdb_path"])
    parser.add_argument("--label_file", type=str, default=DEFAULTS["label_file"])
    parser.add_argument("--train_fasta", type=str, default=DEFAULTS["train_fasta"])
    parser.add_argument("--test_fasta", type=str, nargs="+", default=[DEFAULTS["test_fasta"]])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    wandb.init(
        project="funfam_contrasted",
        name="baseline_knn",
        config={
            "method": "baseline_knn",
            "embedding_dim": 1024,
            "k": args.k,
            "train_fasta": args.train_fasta,
            "test_fasta": args.test_fasta,
        },
    )

    # Load labels
    logger.info("Loading labels…")
    labels, idx_to_label = load_funfam_labels(Path(args.label_file))
    logger.info(f"  {len(labels):,} sequences, {len(idx_to_label):,} classes")

    # Build train dataset & index
    logger.info("Loading train embeddings…")
    train_keys = load_h5_keys_from_fasta(Path(args.train_fasta))
    train_ds = FunfamEmbeddingDataset(
        lmdb_path=Path(args.lmdb_path),
        lmdb_keys=train_keys,
        labels=labels,
    )
    train_emb, train_labels = collect_embeddings(train_ds, args.batch_size)
    logger.info(f"  {train_emb.shape[0]:,} train embeddings ({train_emb.shape[1]}d)")

    train_emb = l2_normalize_np(train_emb)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Evaluate each test set
    test_paths = {}
    for path_str in args.test_fasta:
        test_paths.update(resolve_fasta_paths(Path(path_str)))

    for name, fasta_path in test_paths.items():
        logger.info(f"Evaluating test set '{name}'…")
        test_keys = load_h5_keys_from_fasta(fasta_path)
        test_ds = FunfamEmbeddingDataset(
            lmdb_path=Path(args.lmdb_path),
            lmdb_keys=test_keys,
            labels=labels,
        )
        test_emb, test_labels = collect_embeddings(test_ds, args.batch_size)
        test_emb = l2_normalize_np(test_emb)

        logger.info(f"  searching {test_emb.shape[0]:,} queries (k={args.k})…")
        indices = gpu_search(train_emb, test_emb, args.k, device)
        y_pred = train_labels[indices[:, 0]]

        metrics = compute_metrics(test_labels, y_pred)
        for k, v in metrics.items():
            logger.info(f"  {name} — {k}: {v:.4f}")
            wandb.log({f"test/{name}/knn_{k}": v})

    wandb.finish()


if __name__ == "__main__":
    main()
