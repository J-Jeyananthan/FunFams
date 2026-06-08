"""Data module for funfam training (LMDB embeddings)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import load_h5_keys_from_fasta, load_funfam_labels, resolve_fasta_paths

logger = logging.getLogger(__name__)


def lmdb_key_to_seq_id(lmdb_key: str) -> str:
    """Convert LMDB key (H5-style) to sequence ID format used in mapping file.

    LMDB/H5: Q8EJT8_3-88 -> Q8EJT8/3-88
    LMDB/H5: A0A3Q8W200_25-112_187-234 -> A0A3Q8W200/25-112_187-234
    Mapping: Q8EJT8/3-88 or A0A3Q8W200/25-112_187-234

    Only replaces the first underscore to preserve domain boundaries.
    """
    if "_" in lmdb_key:
        return lmdb_key.replace("_", "/", 1)
    return lmdb_key


class FunfamEmbeddingDataset(Dataset):
    """Funfam protein embeddings from an LMDB (raw float16 buffers)."""

    def __init__(
        self,
        lmdb_path: Path,
        lmdb_keys: List[str],
        labels: Dict[str, int],
        cache_embeddings: bool = False,
        embedding_dim: int = 1024,
        dtype: np.dtype = np.float16,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.embedding_dim = int(embedding_dim)
        self.dtype = dtype

        # NOTE: open LMDB lazily per process
        self._env: Optional[lmdb.Environment] = None

        self._embedding_cache: Optional[Dict[str, torch.Tensor]] = None

        # Filter by labels (fast, in-memory)
        logger.info(f"Filtering {len(lmdb_keys):,} FASTA keys against label map…")
        label_filtered: List[Tuple[str, int]] = []
        missing_labels: List[Tuple[str, str]] = []
        for key in lmdb_keys:
            seq_id = lmdb_key_to_seq_id(key)
            if seq_id in labels:
                label_filtered.append((key, labels[seq_id]))
            else:
                missing_labels.append((key, seq_id))

        logger.info(f"Label filter: {len(label_filtered):,} matched, {len(missing_labels):,} missing labels")

        if missing_labels and len(missing_labels) > len(label_filtered) * 0.1:
            logger.warning(
                f"Warning: {len(missing_labels)}/{len(lmdb_keys)} sequences missing labels. "
                f"First few examples: {missing_labels[:5]}"
            )

        self.samples: List[Tuple[str, int]] = label_filtered

        # Warm OS page cache by reading raw LMDB file sequentially.
        # Much faster than cursor iteration: no per-entry Python overhead,
        # and the OS can prefetch large sequential blocks from SAN.
        data_mdb = Path(self.lmdb_path) / "data.mdb"
        if data_mdb.exists():
            file_size = data_mdb.stat().st_size
            logger.info(
                f"LMDB warmup: reading {file_size / (1024**3):.1f} GB from {data_mdb}…"
            )
            chunk_size = 4 * 1024 * 1024  # 4 MB chunks
            bytes_read = 0
            with open(data_mdb, "rb") as f:
                while True:
                    buf = f.read(chunk_size)
                    if not buf:
                        break
                    bytes_read += len(buf)
                    if bytes_read % (512 * 1024 * 1024) == 0:  # log every 512 MB
                        logger.info(
                            f"  LMDB warmup: {bytes_read / (1024**3):.1f}/{file_size / (1024**3):.1f} GB"
                        )
            logger.info(f"LMDB warmup complete: {bytes_read / (1024**3):.1f} GB read into page cache")

        print(f"✅ cache_embeddings = {cache_embeddings}")

        # Optionally cache embeddings in memory
        if cache_embeddings:
            self._embedding_cache = {}
            with env.begin(write=False) as txn:
                for key, _ in self.samples:
                    buf = txn.get(key.encode("utf-8"))
                    if buf is None:
                        # Should be rare; keep going
                        continue
                    self._embedding_cache[key] = self._decode_embedding(buf)

    def _open_env(self) -> lmdb.Environment:
        """Open an LMDB env (read-only). One handle per process is ideal."""
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            )
        return self._env

    def _decode_embedding(self, buf: bytes) -> torch.Tensor:
        """Decode raw float16 bytes into a torch float32 tensor of shape (embedding_dim,)."""
        arr = np.frombuffer(buf, dtype=self.dtype, count=self.embedding_dim).copy()
        if arr.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding has wrong length: got {arr.shape[0]}, expected {self.embedding_dim}"
            )
        return torch.from_numpy(arr).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        key, label = self.samples[idx]

        if self._embedding_cache is not None:
            return self._embedding_cache[key], label

        env = self._open_env()
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode("utf-8"))
            if buf is None:
                raise KeyError(f"Missing embedding in LMDB for key: {key}")
            return self._decode_embedding(buf), label


class FunfamDataModule(L.LightningDataModule):
    """Funfam protein classification data."""

    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        label_file: str,
        embedding_file: str,  # LMDB directory path
        test_fasta: Union[str, List[str], None] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        cache_embeddings: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if cache_embeddings and num_workers > 0:
            logger.warning(
                "cache_embeddings=True with num_workers>0 may cause memory duplication across workers. "
                "Consider num_workers=0 when caching, or cache_embeddings=False when using multiple workers."
            )

        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta_paths = self._resolve_test_paths(test_fasta)
        self.label_file = Path(label_file)
        self.embedding_file = Path(embedding_file)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()
        self.cache_embeddings = cache_embeddings

        self.labels: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None
        self.test_datasets: Optional[Dict[str, FunfamEmbeddingDataset]] = None

    def _resolve_test_paths(self, test_fasta: Union[str, List[str], None]) -> Dict[str, Path]:
        if test_fasta is None:
            return {}
        if isinstance(test_fasta, list):
            return {Path(p).stem: Path(p) for p in test_fasta}
        return resolve_fasta_paths(Path(test_fasta))

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def setup(self, stage: Optional[str] = None):
        if self.labels is None:
            self.labels, self.idx_to_label = load_funfam_labels(self.label_file)

        if stage in ("fit", "test", None):
            if not hasattr(self, "train_dataset"):
                self.train_dataset = self._create_dataset(self.train_fasta)

        if stage in ("fit", None):
            self.val_dataset = self._create_dataset(self.val_fasta)

        if stage in ("test", None):
            self.test_datasets = {
                name: self._create_dataset(path)
                for name, path in self.test_fasta_paths.items()
            }
            if self.test_datasets:
                self.test_dataset = next(iter(self.test_datasets.values()))

    def _create_dataset(self, fasta_path: Path) -> FunfamEmbeddingDataset:
        return FunfamEmbeddingDataset(
            lmdb_path=self.embedding_file,
            lmdb_keys=load_h5_keys_from_fasta(fasta_path),
            labels=self.labels,
            cache_embeddings=self.cache_embeddings,
            embedding_dim=1024,
            dtype=np.float16,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.test_datasets) == 1:
            return self._dataloader(self.test_dataset)
        return [self._dataloader(ds) for ds in self.test_datasets.values()]

    def get_test_dataloader(self, name: str) -> Optional[DataLoader]:
        if self.test_datasets and name in self.test_datasets:
            return self._dataloader(self.test_datasets[name])
        return None

    def get_test_names(self) -> List[str]:
        return list(self.test_fasta_paths.keys())

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
