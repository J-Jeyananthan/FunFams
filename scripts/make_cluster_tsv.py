#!/usr/bin/env python3
"""Build a rep→member TSV from a directory of .faa cluster files.

Each .faa file is one cluster; the first sequence is the representative.
Subdirectories are walked recursively.

Output TSV format (same as MMseqs2 cluster TSV):
    rep_id<TAB>member_id
    (rep is included as a member of itself)

Example:
    python make_cluster_tsv.py \
        --faa_dir        .../superfamily_clusters/ \
        --output         .../cluster.tsv \
        --reps_fasta     .../cluster_reps.fasta \
        --rep_ids        .../cluster_rep_ids.txt \
        --all_seqs_fasta .../all_seqs.fasta
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build rep→member TSV from .faa cluster files.")
    parser.add_argument("--faa_dir",       required=True, help="Root directory containing .faa cluster files")
    parser.add_argument("--output",        required=True, help="Output TSV file path")
    parser.add_argument("--reps_fasta",    required=True, help="Output FASTA file of rep sequences")
    parser.add_argument("--rep_ids",       required=True, help="Output text file of rep IDs (one per line)")
    parser.add_argument("--all_seqs_fasta",required=True, help="Output FASTA file of all sequences across all .faa files")
    args = parser.parse_args()

    faa_files = sorted(Path(args.faa_dir).rglob("*.faa"))
    print(f"Found {len(faa_files)} .faa files")

    n_clusters = 0
    n_members = 0

    with open(args.output, "w") as out, \
         open(args.reps_fasta, "w") as reps_fh, \
         open(args.rep_ids, "w") as rep_ids_fh, \
         open(args.all_seqs_fasta, "w") as all_fh:

        for faa_path in faa_files:
            rep = None
            rep_seq_lines = []
            members = []
            capturing_rep = False

            with open(faa_path) as f:
                for line in f:
                    all_fh.write(line)
                    if line.startswith(">"):
                        seq_id = line[1:].strip()
                        if rep is None:
                            rep = seq_id
                            capturing_rep = True
                        else:
                            capturing_rep = False
                        members.append(seq_id)
                    elif capturing_rep:
                        rep_seq_lines.append(line.rstrip())

            if rep is None or not members:
                continue

            for member in members:
                out.write(f"{rep}\t{member}\n")

            reps_fh.write(f">{rep}\n")
            reps_fh.write("\n".join(rep_seq_lines) + "\n")
            rep_ids_fh.write(f"{rep}\n")

            n_clusters += 1
            n_members += len(members)

    print(f"Written {n_clusters} clusters, {n_members} members → {args.output}")
    print(f"Written {n_clusters} rep sequences → {args.reps_fasta}")
    print(f"Written {n_clusters} rep IDs → {args.rep_ids}")
    print(f"Written {n_members} sequences → {args.all_seqs_fasta}")


if __name__ == "__main__":
    main()
