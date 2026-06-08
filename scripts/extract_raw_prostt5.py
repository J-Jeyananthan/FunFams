"""Extract raw ProstT5 embeddings from LMDB and save in .pt format for create_distance_matrix.py.

Usage:
    python scripts/extract_raw_prostt5.py \
        --lmdb_path path/to/embeddings.lmdb \
        --fasta path/to/sequences.fasta \
        --output_dir path/to/output/
"""

import argparse
from pathlib import Path

import lmdb
import numpy as np
import torch


def parse_fasta_headers(fasta_path: Path) -> list[tuple[str, str]]:
    """Return (fasta_id, lmdb_key) pairs in FASTA order."""
    pairs = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                fasta_id = line.strip().lstrip(">")
                lmdb_key = fasta_id.replace("/", "_", 1)
                pairs.append((fasta_id, lmdb_key))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract raw ProstT5 embeddings from LMDB.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB directory")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file (defines order and IDs)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_fasta_headers(Path(args.fasta))
    print(f"FASTA: {len(pairs)} sequences")

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
    pt_data = []
    missing = []
    with env.begin(write=False) as txn:
        for fasta_id, key in pairs:
            buf = txn.get(key.encode("utf-8"))
            if buf is None:
                missing.append(fasta_id)
                continue
            arr = np.frombuffer(buf, dtype=np.float16, count=1024).copy().astype(np.float32)
            arr = arr / np.linalg.norm(arr)
            pt_data.append({"label": fasta_id, "mean_representations": {33: torch.from_numpy(arr)}})
    env.close()

    if missing:
        print(f"WARNING: {len(missing)} sequences not found in LMDB. First 5: {missing[:5]}")

    torch.save(pt_data, output_dir / "E_starting_prostt5.pt")
    print(f"Saved E_starting_prostt5.pt ({len(pt_data)} embeddings, 1024-d) to {output_dir}")


if __name__ == "__main__":
    main()
