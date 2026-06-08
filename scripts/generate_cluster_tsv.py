#!/usr/bin/env python3
"""
Generates from a directory of subdirectories each containing .faa files,
where the first sequence in each .faa file is the cluster rep:
  1. A MMseqs2-style cluster TSV (rep_id\tmember_id)
  2. A list of cluster rep IDs
  3. A FASTA file containing only the cluster rep sequences

Usage: python generate_cluster_tsv.py <clusters_dir> <tsv_path> <rep_ids_path> <rep_fasta_path>
"""

import sys
from pathlib import Path

def process_faa(faa_path):
    rep_id = None
    members = []
    rep_seq_lines = []
    in_rep = False

    with open(faa_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                members.append(seq_id)
                if rep_id is None:
                    rep_id = seq_id
                    in_rep = True
                    rep_seq_lines.append(line)
                else:
                    in_rep = False
            elif in_rep:
                rep_seq_lines.append(line)

    return rep_id, members, rep_seq_lines

def main():
    clusters_dir = Path(sys.argv[1])
    tsv_path = Path(sys.argv[2])
    rep_ids_path = Path(sys.argv[3])
    rep_fasta_path = Path(sys.argv[4])

    faa_files = sorted(clusters_dir.glob("*/*.faa"))

    with open(tsv_path, 'w') as tsv, open(rep_ids_path, 'w') as rep_ids, open(rep_fasta_path, 'w') as rep_fasta:
        for faa_path in faa_files:
            rep_id, members, rep_seq_lines = process_faa(faa_path)
            if rep_id is None:
                continue
            for member in members:
                tsv.write(f"{rep_id}\t{member}\n")
            rep_ids.write(f"{rep_id}\n")
            rep_fasta.write('\n'.join(rep_seq_lines) + '\n')

if __name__ == '__main__':
    main()
