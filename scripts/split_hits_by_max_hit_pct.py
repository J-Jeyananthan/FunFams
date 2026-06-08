"""Split a hits TSV by max sequence identity thresholds (20–90%)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split hits TSV into subsets by max identity pct (<=20, <=30, ..., <=90).",
    )
    parser.add_argument("input_tsv", type=Path, help="Input hits TSV (from seq_sim_split.py).")
    parser.add_argument("output_dir", type=Path, help="Directory to write split TSV files.")
    args = parser.parse_args()

    if not args.input_tsv.exists():
        raise FileNotFoundError(f"Missing: {args.input_tsv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = args.input_tsv.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]

    thresholds = [20, 30, 40, 50, 60, 70, 80, 90]

    for threshold in thresholds:
        kept = []
        for row in rows:
            parts = row.split("\t")
            pct = parts[1].strip()
            if pct == "NA" or float(pct) <= threshold:
                kept.append(row)
        out_file = args.output_dir / f"test_le{threshold}.tsv"
        out_file.write_text("\n".join([header] + kept) + "\n", encoding="utf-8")
        print(f"le{threshold}: {len(kept)} sequences -> {out_file}")


if __name__ == "__main__":
    main()
