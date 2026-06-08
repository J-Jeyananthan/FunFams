#!/usr/bin/env python3
"""Split HUPs sequences from deduplicated FASTA into train/val sets (95/5 random split)."""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Random 95/5 train/val split for HUPs sequences.")
    parser.add_argument("--fasta",      required=True, help="Deduplicated input FASTA")
    parser.add_argument("--ids",        required=True, help="File with HUPs sequence IDs (one per line)")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/val FASTAs")
    parser.add_argument("--val_frac",   type=float, default=0.05, help="Fraction for val set (default: 0.05)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = set(Path(args.ids).read_text().splitlines())
    print(f"Loaded {len(ids):,} HUPs IDs")

    train_ids, val_ids = train_test_split(
        list(ids), test_size=args.val_frac, random_state=args.seed
    )
    train_ids = set(train_ids)
    val_ids = set(val_ids)
    print(f"Split: {len(train_ids):,} train / {len(val_ids):,} val")

    current_out = None
    n_train = n_val = 0

    with open(args.fasta) as f, \
         open(out_dir / "hups_train.fasta", "w") as train_fh, \
         open(out_dir / "hups_val.fasta",   "w") as val_fh:

        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip()
                if seq_id in train_ids:
                    current_out = train_fh
                    n_train += 1
                elif seq_id in val_ids:
                    current_out = val_fh
                    n_val += 1
                else:
                    current_out = None
            if current_out:
                current_out.write(line)

    print(f"Written: {n_train:,} train -> {out_dir}/hups_train.fasta")
    print(f"Written: {n_val:,} val   -> {out_dir}/hups_val.fasta")


if __name__ == "__main__":
    main()
