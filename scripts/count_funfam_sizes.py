"""
Count sequences per .faa file in a directory of FunFam clusters.
Outputs the number of files and sequence counts, suitable for plotting cluster size distribution.
"""

import argparse
import sys
from pathlib import Path


def count_sequences(faa_path: Path) -> int:
    count = 0
    with open(faa_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Count sequences per FunFam .faa file")
    parser.add_argument("directory", type=Path, help="Directory containing .faa files")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional output TSV file (default: print to stdout)"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    faa_files = sorted([f for ext in ("*.faa", "*.aln") for f in args.directory.glob(ext)])
    if not faa_files:
        print(f"No .faa or .aln files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    counts = [(f.name, count_sequences(f)) for f in faa_files]

    header = f"num_files\t{len(counts)}"
    rows = ["funfam\tnum_sequences"] + [f"{name}\t{n}" for name, n in counts]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as out:
            out.write(header + "\n")
            out.write("\n".join(rows) + "\n")
        print(f"Wrote {len(counts)} entries to {args.output}")
    else:
        print(header)
        print("\n".join(rows))


if __name__ == "__main__":
    main()
