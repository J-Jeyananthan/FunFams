"""Randomly split a deduplicated FASTA file into train and validation sets.

Performs the 95/5 random split described in the Methods section of the report.
Input should be a deduplicated FASTA (e.g. output of seqkit rmdup).

Usage:
    python scripts/datasplit.py \
        --input deduplicated.fasta \
        --output_dir splits/ \
        --val_size 0.05 \
        --seed 42
"""

import argparse
import random
from pathlib import Path


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) pairs from a FASTA file."""
    records = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w") as f:
        for header, seq in records:
            f.write(header + "\n" + seq + "\n")


def main():
    parser = argparse.ArgumentParser(description="Random train/val split of a FASTA file.")
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA file (deduplicated).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write train.fasta and val.fasta.")
    parser.add_argument("--val_size", type=float, default=0.05, help="Fraction of sequences for validation (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    records = parse_fasta(args.input)
    random.seed(args.seed)
    random.shuffle(records)

    n_val = int(len(records) * args.val_size)
    val, train = records[:n_val], records[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(train, args.output_dir / "train.fasta")
    write_fasta(val, args.output_dir / "val.fasta")

    print(f"Train: {len(train)} sequences -> {args.output_dir / 'train.fasta'}")
    print(f"Val:   {len(val)} sequences -> {args.output_dir / 'val.fasta'}")


if __name__ == "__main__":
    main()
