"""Extract 128-d projected embeddings from a trained checkpoint.

Usage:
    python scripts/extract_embeddings.py \
        --checkpoint path/to/checkpoint.ckpt \
        --lmdb_path path/to/embeddings.lmdb \
        --fasta path/to/sequences.fasta \
        --output_dir path/to/output/
"""

import argparse
from pathlib import Path

import lmdb
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import FunfamSupConModel


def parse_fasta_headers(fasta_path: Path) -> list[tuple[str, str]]:
    """Return (fasta_id, lmdb_key) pairs in FASTA order.

    fasta_id: exact text after '>' (slash format, e.g. Q8EJT8/3-88)
    lmdb_key: underscore format used as LMDB key (e.g. Q8EJT8_3-88)
    """
    pairs = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                fasta_id = line.strip().lstrip(">")
                lmdb_key = fasta_id.replace("/", "_", 1)
                pairs.append((fasta_id, lmdb_key))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract projected embeddings from a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB directory")
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file (defines order and IDs)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for E.npy and reps_ids.txt")
    parser.add_argument("--batch_size", type=int, default=2048, help="Inference batch size")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Raw embedding dimension in LMDB (1024 for ProstT5, 1280 for ESM2-650M)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (projection head only, no loss needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FunfamSupConModel.load_from_checkpoint(args.checkpoint, map_location=device, weights_only=False, strict=False)
    model.eval()
    model.to(device)

    # Parse FASTA for ordered IDs
    pairs = parse_fasta_headers(Path(args.fasta))
    fasta_ids = [fasta_id for fasta_id, _ in pairs]
    lmdb_keys = [lmdb_key for _, lmdb_key in pairs]
    print(f"FASTA: {len(fasta_ids)} sequences")

    # Read embeddings from LMDB
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
    raw_embeddings = []
    missing = []
    with env.begin(write=False) as txn:
        for fasta_id, key in pairs:
            buf = txn.get(key.encode("utf-8"))
            if buf is None:
                missing.append(fasta_id)
                continue
            arr = np.frombuffer(buf, dtype=np.float16, count=args.embedding_dim).copy()
            raw_embeddings.append(arr)
    env.close()

    if missing:
        print(f"WARNING: {len(missing)} sequences not found in LMDB. First 5: {missing[:5]}")
        # Remove missing from ID list
        found_set = set(missing)
        fasta_ids = [fid for fid in fasta_ids if fid not in found_set]

    print(f"Loaded {len(raw_embeddings)} embeddings from LMDB")

    # Project through model in batches
    raw_tensor = torch.from_numpy(np.stack(raw_embeddings)).float()
    all_projected = []

    with torch.no_grad():
        for i in range(0, len(raw_tensor), args.batch_size):
            batch = raw_tensor[i : i + args.batch_size].to(device)
            proj = model(batch)
            all_projected.append(proj.cpu())

    E = torch.cat(all_projected, dim=0).numpy().astype(np.float32)
    assert E.shape == (len(fasta_ids), 128), f"Shape mismatch: {E.shape} vs ({len(fasta_ids)}, 128)"

    # Save outputs
    np.save(output_dir / "E_starting.npy", E)
    with open(output_dir / "starting_ids_3698.txt", "w") as f:
        for seq_id in fasta_ids:
            f.write(seq_id + "\n")

    pt_data = [{"label": fasta_ids[i], "mean_representations": {33: torch.from_numpy(E[i])}} for i in range(len(fasta_ids))]
    torch.save(pt_data, output_dir / "E_starting_contrastive.pt")

    print(f"Saved E_starting.npy {E.shape}, starting_ids_3698.txt, and E_starting_contrastive.pt ({len(fasta_ids)} IDs) to {output_dir}")


if __name__ == "__main__":
    main()
