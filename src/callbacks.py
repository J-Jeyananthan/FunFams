"""
Streaming k-NN evaluation callback (test-only) for FunFam training.
"""

import logging
from typing import Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

logger = logging.getLogger(__name__)


# -------------------------
# Utilities
# -------------------------
def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=np.union1d(y_true, y_pred)),
    }


# -------------------------
# Callback
# -------------------------
class KNNEvaluationCallback(L.Callback):
    """
    Streaming 1-NN evaluation (test-only) using PyTorch matmul search.

    - Streams train embeddings into a numpy array
    - Accumulates test queries then does a single chunked search
    - Search uses torch.matmul (MKL) — no FAISS OMP conflict
    """

    def __init__(
        self,
        batch_size_override: Optional[int] = None,
        faiss_threads: Optional[int] = None,  # kept for Hydra compatibility, unused
        max_train_points: Optional[int] = None,
        eval_every_n_epochs: Optional[int] = None,  # ignored, kept for Hydra compatibility
    ):
        super().__init__()
        self.batch_size_override = batch_size_override
        self.max_train_points = max_train_points

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.world_size > 1 and not trainer.is_global_zero:
            return

        # Handle multiple test sets
        if hasattr(trainer.datamodule, "test_datasets") and trainer.datamodule.test_datasets:
            test_sets = list(trainer.datamodule.test_datasets.items())
        else:
            test_sets = [("", trainer.datamodule.test_dataset)]

        try:
            train_ds = trainer.datamodule.train_dataset
            bs = int(self.batch_size_override or trainer.datamodule.hparams.batch_size)

            logger.info("🔎 [kNN-test] Building train index (streaming train set)…")
            train_vecs, train_labels = self._build_index(trainer, pl_module, train_ds, bs)

            for test_name, test_ds in test_sets:
                logger.info(f"🔎 [kNN-test] Evaluating test set '{test_name or 'default'}'…")
                metrics = self._eval_test(trainer, pl_module, train_vecs, train_labels, test_ds, bs)

                prefix = f"test/{test_name}/" if test_name else "test/"
                for k, v in metrics.items():
                    pl_module.log(
                        f"{prefix}knn_{k}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=False,
                    )

                logger.info(f"✅ [kNN-test] {test_name or 'default'} metrics: {metrics}")

        except Exception:
            logger.exception("Error during streaming kNN test evaluation")

    # -------------------------
    # Internal helpers
    # -------------------------
    @torch.no_grad()
    def _build_index(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        dataset,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from torch.utils.data import DataLoader

        pl_module.eval()
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        vectors = []
        labels = []
        total = 0

        for i, (emb, lab) in enumerate(dl):
            proj = pl_module(emb.to(pl_module.device, non_blocking=True))
            x = proj.detach().float().cpu().numpy().astype(np.float32, copy=False)
            x = l2_normalize_np(x)

            vectors.append(x)
            labels.append(lab.cpu().numpy().astype(np.int64, copy=False))

            total += x.shape[0]
            if self.max_train_points and total >= self.max_train_points:
                break

            if i > 0 and i % 500 == 0:
                logger.info(f"  indexed {total:,} train vectors…")

        train_vecs = np.concatenate(vectors, axis=0)[:total]
        train_labels = np.concatenate(labels, axis=0)[:total]

        logger.info(f"  train index ready with {total:,} vectors")
        return train_vecs, train_labels

    @torch.no_grad()
    def _eval_test(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        train_vecs: np.ndarray,
        train_labels: np.ndarray,
        dataset,
        batch_size: int,
    ) -> Dict[str, float]:
        from torch.utils.data import DataLoader

        pl_module.eval()
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Accumulate all test queries first, then bulk search.
        all_queries = []
        all_labels = []
        seen = 0

        for i, (emb, lab) in enumerate(dl):
            proj = pl_module(emb.to(pl_module.device, non_blocking=True))
            q = proj.detach().float().cpu().numpy().astype(np.float32, copy=False)
            q = l2_normalize_np(q)
            all_queries.append(q)
            all_labels.append(lab.cpu().numpy().astype(np.int64, copy=False))
            seen += lab.shape[0]

            if i > 0 and i % 500 == 0:
                logger.info(f"  projected {seen:,} test samples…")

        all_queries = np.concatenate(all_queries, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        n_train = train_vecs.shape[0]
        logger.info(f"  searching {all_queries.shape[0]:,} queries against {n_train:,} train vectors…")

        device = pl_module.device

        # GPU matmul search: uses the same device the model ran on.
        # GPU memory bandwidth is ~7x higher than DDR, making this much faster
        # than CPU matmul for this memory-bandwidth-bound operation.
        db_t = torch.from_numpy(train_vecs).to(device)   # (N, D)

        # Chunk queries based on free memory after db_t is loaded.
        if device.type == "cuda":
            free_mem = torch.cuda.mem_get_info(device)[0]
            target_bytes = int(free_mem * 0.50)
        else:
            target_bytes = 1024**3
        chunk_q = max(1, min(len(all_queries), int(target_bytes // (n_train * 4))))
        logger.info(f"  query chunk size: {chunk_q} (device: {device})")

        nn_idx = torch.empty(len(all_queries), dtype=torch.long)
        for q0 in range(0, len(all_queries), chunk_q):
            q1 = min(q0 + chunk_q, len(all_queries))
            q_chunk = torch.from_numpy(all_queries[q0:q1]).to(device)
            scores = q_chunk @ db_t.T   # (chunk_q, N)
            nn_idx[q0:q1] = scores.argmax(dim=1).cpu()
            del q_chunk, scores

        y_pred = train_labels[nn_idx.numpy()]
        return compute_metrics(all_labels, y_pred)
