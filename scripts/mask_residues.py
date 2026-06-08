#!/usr/bin/env python3
"""Mask a percentage of amino acid residues in a FASTA file with 'X'.

Generates noisy FASTA files for robustness experiments.
After generating, re-embed with embed-david.py to create a new LMDB,
then train with the masked LMDB to measure performance degradation.

Usage:
    # Single noise level
    python scripts/mask_residues.py -i train.fasta -p 10 -o train_masked_10.fasta

    # Multiple noise levels (generates one file per level)
    python scripts/mask_residues.py -i train.fasta -p 5 10 20 50

    # Custom seed for reproducibility
    python scripts/mask_residues.py -i train.fasta -p 15 --seed 123
"""

import argparse
import random
from pathlib import Path


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def mask_sequence(seq: str, mask_pct: float, rng: random.Random) -> tuple[str, int]:
    """Replace mask_pct% of standard amino acid positions with 'X'.

    Returns the masked sequence and the number of residues masked.
    """
    seq_list = list(seq)
    # Only mask standard amino acid positions (skip existing X, gaps, etc.)
    maskable = [i for i, c in enumerate(seq_list) if c in STANDARD_AA]
    n_mask = round(len(maskable) * mask_pct / 100)
    positions = rng.sample(maskable, min(n_mask, len(maskable)))
    for i in positions:
        seq_list[i] = "X"
    return "".join(seq_list), len(positions)


def mask_fasta(input_path: Path, output_path: Path, mask_pct: float, seed: int):
    """Read a FASTA file and write a masked version."""
    rng = random.Random(seed)
    total_seqs = 0
    total_residues = 0
    total_masked = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        header = None
        seq_chunks = []

        def flush():
            nonlocal total_seqs, total_residues, total_masked
            if header is None:
                return
            seq = "".join(seq_chunks)
            masked_seq, n_masked = mask_sequence(seq, mask_pct, rng)
            total_seqs += 1
            total_residues += len(seq)
            total_masked += n_masked
            fout.write(header + "\n")
            # Write sequence in 60-char lines (standard FASTA)
            for i in range(0, len(masked_seq), 60):
                fout.write(masked_seq[i : i + 60] + "\n")

        for line in fin:
            if line.startswith(">"):
                flush()
                header = line.rstrip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())

        flush()

    actual_pct = (total_masked / max(total_residues, 1)) * 100
    print(f"  {output_path.name}: {total_seqs} sequences, "
          f"{total_masked}/{total_residues} residues masked ({actual_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Mask residues in a FASTA file with 'X' for noise injection experiments."
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Input FASTA file path.")
    parser.add_argument("-p", "--percentages", required=True, type=float, nargs="+",
                        help="Masking percentage(s), e.g. 5 10 20 50.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output FASTA path (only valid with a single percentage). "
                             "If not given, auto-generates names like input_masked_10.fasta.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output and len(args.percentages) > 1:
        parser.error("-o/--output can only be used with a single percentage.")

    print(f"Input: {input_path}")
    print(f"Seed: {args.seed}")
    print()

    for pct in args.percentages:
        if not 0 < pct < 100:
            print(f"  Skipping invalid percentage: {pct} (must be between 0 and 100)")
            continue

        if args.output:
            output_path = Path(args.output)
        else:
            stem = input_path.stem
            suffix = input_path.suffix
            tag = f"{pct:g}"  # e.g. "10" not "10.0"
            output_path = input_path.parent / f"{stem}_masked_{tag}{suffix}"

        mask_fasta(input_path, output_path, pct, args.seed)

    print("\nDone. Next steps:")
    print("  1. Re-embed each masked FASTA with embed-david.py to create new LMDB(s)")
    print("  2. Train with: python src/train.py paths.train_fasta=<masked.fasta> "
          "paths.embedding_file=<masked.lmdb>")


if __name__ == "__main__":
    main()
