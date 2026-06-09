"""Utility functions for funfam training."""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def fasta_to_h5_key(header: str) -> str:
    """Convert FASTA header to LMDB key format.

    FASTA: >A0A151J9E7/282-311
    LMDB:  A0A151J9E7_282-311
    """
    return header.strip().lstrip('>').replace('/', '_')


def load_h5_keys_from_fasta(fasta_path: Path) -> List[str]:
    """Read FASTA file and return list of LMDB keys."""
    h5_keys = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                try:
                    h5_keys.append(fasta_to_h5_key(line))
                except ValueError as e:
                    logger.warning(f"Could not parse header: {line.strip()} - {e}")
    return h5_keys


def load_funfam_labels(mapping_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load funfam labels from mapping file.
    
    File format: {sequence_id}\t{funfam_id}
    Returns: (sequence_id -> funfam_idx, funfam_idx -> funfam_id)
    """
    id_to_ff_idx: Dict[str, int] = {}
    ff_to_idx: Dict[str, int] = {}
    
    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                seq_id, funfam_id = parts[0], parts[1]
                if funfam_id not in ff_to_idx:
                    ff_to_idx[funfam_id] = len(ff_to_idx)
                id_to_ff_idx[seq_id] = ff_to_idx[funfam_id]
    
    return id_to_ff_idx, {v: k for k, v in ff_to_idx.items()}


def resolve_fasta_paths(fasta_input: Path) -> Dict[str, Path]:
    """Resolve FASTA paths - handles single file or directory."""
    if fasta_input.is_dir():
        fasta_files = sorted(fasta_input.glob("*.fasta")) + sorted(fasta_input.glob("*.fa"))
        if not fasta_files:
            logger.warning(f"No FASTA files found in directory: {fasta_input}")
            return {}
        logger.info(f"Found {len(fasta_files)} FASTA files in {fasta_input}")
        return {f.stem: f for f in fasta_files}
    if fasta_input.is_file():
        return {fasta_input.stem: fasta_input}
    logger.warning(f"FASTA path not found: {fasta_input}")
    return {}
