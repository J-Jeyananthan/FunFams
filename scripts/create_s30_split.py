#!/usr/bin/env python3
"""
Create train/val/test splits from FunFam dataset using S30 clustering.

Pipeline:
1. Cluster input sequences at S30 (assumes input is already deduplicated)
2. Eligible clusters: rep's FunFam has >= 3 sequences
3. Shuffle; assign first test_size clusters to test, next val_size to val; take cluster rep
4. Train = all remaining sequences (non-reps from holdout clusters + all train cluster seqs)
5. Remove train sequences with >= min_seq_id similarity to val/test via MMseqs2 search
6. Drop val/test sequences whose FunFam has no training examples
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_fasta(file_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_header: str | None = None
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:]
                sequences[current_header] = ""
            elif current_header is not None:
                sequences[current_header] += line
    return sequences


def extract_domain_id(header: str) -> str:
    header = header.lstrip(">").split()[0]
    if "|" in header:
        domain_part = header.split("|")[-1]
        if "/" in domain_part:
            return domain_part.split("/")[0]
        if "_" in domain_part:
            parts = domain_part.split("_")
            for part in parts:
                if "-" not in part or not all(c.isdigit() or c == "-" for c in part):
                    return part
            return parts[0]
        return domain_part
    return header


def load_assignments(mapping_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            domain_id, funfam = line.split("\t")
            mapping[domain_id] = funfam
    return mapping


def write_fasta(file_path: Path, sequences: dict[str, str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n{seq}\n")


def write_domain_id_fasta(file_path: Path, sequences: dict[str, str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for header, seq in sequences.items():
            domain_id = extract_domain_id(header)
            if not domain_id:
                continue
            f.write(f">{domain_id}\n{seq}\n")


def run_mmseqs2_cluster(
    input_fasta: Path,
    output_prefix: Path,
    tmp_dir: Path,
    *,
    min_seq_id: float,
    sensitivity: float,
    coverage: float,
) -> tuple[Path, Path]:
    """Run mmseqs easy-cluster. Returns (rep_seq_fasta, cluster_tsv)."""
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 not found in PATH.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-cluster",
        str(input_fasta),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "-s", str(sensitivity),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 easy-cluster failed (exit {result.returncode}) — see output above")

    rep_fasta = Path(str(output_prefix) + "_rep_seq.fasta")
    cluster_tsv = Path(str(output_prefix) + "_cluster.tsv")
    if not rep_fasta.exists() or not cluster_tsv.exists():
        raise RuntimeError(f"MMseqs2 easy-cluster output not found at {output_prefix}_*")
    return rep_fasta, cluster_tsv


def run_mmseqs2_search(
    query_fasta: Path,
    target_fasta: Path,
    *,
    min_seq_id: float,
    sensitivity: float,
    coverage: float,
    num_iterations: int,
) -> set[str]:
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 not found in PATH.")

    tmpdir = Path(tempfile.mkdtemp())
    query_db = tmpdir / "queryDB"
    target_db = tmpdir / "targetDB"
    result_db = tmpdir / "resultDB"
    alignments_file = tmpdir / "alignments.tsv"

    try:
        subprocess.run(
            ["mmseqs", "createdb", str(query_fasta), str(query_db)],
            check=True,
        )
        subprocess.run(
            ["mmseqs", "createdb", str(target_fasta), str(target_db)],
            check=True,
        )
        search_cmd = [
            "mmseqs", "search",
            str(query_db), str(target_db), str(result_db),
            str(tmpdir / "tmp"),
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-s", str(sensitivity),
            "--num-iterations", str(num_iterations),
            "--max-seqs", "1000000",
            "-a",
        ]
        result = subprocess.run(search_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 search failed (exit {result.returncode}) — see output above")

        convert_cmd = [
            "mmseqs", "convertalis",
            str(query_db), str(target_db), str(result_db),
            str(alignments_file),
            "--format-output", "query,target,fident",
        ]
        result = subprocess.run(convert_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 convertalis failed (exit {result.returncode}) — see output above")

        matching_targets: set[str] = set()
        with open(alignments_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and float(parts[2]) >= min_seq_id:
                    matching_targets.add(parts[1])
        return matching_targets
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_cluster_tsv(cluster_tsv: Path) -> dict[str, list[str]]:
    """Load MMseqs2 cluster TSV. Returns {rep_seqid: [member_seqids...]}."""
    clusters: dict[str, list[str]] = {}
    with open(cluster_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")
            clusters.setdefault(rep, []).append(member)
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create S30-based train/val/test split from FunFam dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA (pre-deduplicated sequences)")
    parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="Tab-separated domain_id -> FunFam file (e.g. 'Q8EJT8/3-88\\t1.10.10.10-FF-000001')",
    )
    parser.add_argument("--output_dir", type=Path, default="data/s30_split")
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Number of FunFams to hold out for test. Omit to report eligible FunFam count and exit.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=None,
        help="Number of FunFams to hold out for val. Omit to report eligible FunFam count and exit.",
    )
    parser.add_argument("--min_seq_id", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--mmseqs_sensitivity", type=float, default=7.5)
    parser.add_argument("--mmseqs_coverage", type=float, default=0.8)
    parser.add_argument("--mmseqs_iterations", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    rng = random.Random(args.random_seed)

    if not args.fasta.exists():
        raise FileNotFoundError(args.fasta)
    if not args.assignments.exists():
        raise FileNotFoundError(args.assignments)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading assignments...")
    ff_mapping = load_assignments(args.assignments)

    dedup_sequences = parse_fasta(args.fasta)
    logger.info("Loaded %d sequences", len(dedup_sequences))

    # Step 2: Cluster at S30 (skip if outputs already exist)
    s30_cluster_tsv = args.output_dir / "s30_cluster.tsv"
    if s30_cluster_tsv.exists():
        logger.info("Found existing S30 cluster output, skipping clustering.")
    else:
        logger.info("Clustering at S30...")
        _, s30_cluster_tsv = run_mmseqs2_cluster(
            args.fasta,
            args.output_dir / "s30",
            args.output_dir / "mmseqs_tmp" / "s30",
            min_seq_id=args.min_seq_id,
            sensitivity=args.mmseqs_sensitivity,
            coverage=args.mmseqs_coverage,
        )
    clusters = load_cluster_tsv(s30_cluster_tsv)
    logger.info("S30 clusters: %d", len(clusters))

    # Count FunFam sizes from deduplicated sequences
    ff_counts: Counter[str] = Counter()
    for header in dedup_sequences:
        domain_id = extract_domain_id(header)
        ff = ff_mapping.get(domain_id)
        if ff:
            ff_counts[ff] += 1

    # Build seqid -> full header lookup (cluster TSV uses first word of header as seqid)
    header_by_seqid: dict[str, str] = {h.split()[0]: h for h in dedup_sequences}

    # Step 3: Group eligible cluster reps by FunFam (>= 3 sequences in FunFam)
    ff_to_clusters: dict[str, list[str]] = {}
    for rep_seqid in clusters:
        rep_header = header_by_seqid.get(rep_seqid, rep_seqid)
        domain_id = extract_domain_id(rep_header)
        ff = ff_mapping.get(domain_id)
        if ff and ff_counts.get(ff, 0) >= 3:
            ff_to_clusters.setdefault(ff, []).append(rep_seqid)

    eligible_ffs = list(ff_to_clusters.keys())
    logger.info("Eligible FunFams: %d", len(eligible_ffs))

    if args.test_size is None or args.val_size is None:
        logger.info(
            "No --test_size/--val_size provided. "
            "Rerun with --test_size and --val_size to create splits. "
            "Clustering outputs are cached in %s.",
            args.output_dir,
        )
        return

    rng.shuffle(eligible_ffs)

    if len(eligible_ffs) < args.test_size + args.val_size:
        raise ValueError(
            f"Not enough eligible FunFams ({len(eligible_ffs)}) "
            f"for test+val ({args.test_size + args.val_size})"
        )

    # Step 4: Assign FunFams to test/val pools, pick one cluster rep per FunFam
    test_ffs = eligible_ffs[: args.test_size]
    val_ffs = eligible_ffs[args.test_size : args.test_size + args.val_size]

    test_set: dict[str, str] = {}
    for ff in test_ffs:
        seqid = rng.choice(ff_to_clusters[ff])
        header = header_by_seqid.get(seqid, seqid)
        test_set[header] = dedup_sequences[header]

    val_set: dict[str, str] = {}
    for ff in val_ffs:
        seqid = rng.choice(ff_to_clusters[ff])
        header = header_by_seqid.get(seqid, seqid)
        val_set[header] = dedup_sequences[header]

    logger.info("Selected test/val: %d / %d", len(test_set), len(val_set))

    # Step 5: Train = all sequences not selected as val/test reps
    holdout_headers = set(test_set.keys()) | set(val_set.keys())
    train_set = {h: s for h, s in dedup_sequences.items() if h not in holdout_headers}
    logger.info("Initial train size: %d", len(train_set))

    # Step 6: Remove train sequences with >= min_seq_id similarity to val/test
    logger.info("Running MMseqs2 identity filtering...")
    test_val_fasta = args.output_dir / "temp_test_val.fasta"
    train_temp_fasta = args.output_dir / "temp_train.fasta"
    write_domain_id_fasta(test_val_fasta, {**test_set, **val_set})
    write_domain_id_fasta(train_temp_fasta, train_set)

    matching_train_ids = run_mmseqs2_search(
        test_val_fasta,
        train_temp_fasta,
        min_seq_id=args.min_seq_id,
        sensitivity=args.mmseqs_sensitivity,
        coverage=args.mmseqs_coverage,
        num_iterations=args.mmseqs_iterations,
    )

    test_val_ids = {extract_domain_id(h) for h in holdout_headers}
    remove_ids = matching_train_ids | test_val_ids

    train_set = {
        h: s for h, s in train_set.items()
        if extract_domain_id(h) not in remove_ids
    }
    logger.info(
        "Train size after filtering: %d (removed %d by identity)",
        len(train_set), len(matching_train_ids),
    )

    # Step 7: Drop val/test whose FunFam has no training examples
    train_ffs = {
        ff_mapping.get(extract_domain_id(h))
        for h in train_set
        if ff_mapping.get(extract_domain_id(h))
    }

    filtered_test = {
        h: s for h, s in test_set.items()
        if ff_mapping.get(extract_domain_id(h)) in train_ffs
    }
    filtered_val = {
        h: s for h, s in val_set.items()
        if ff_mapping.get(extract_domain_id(h)) in train_ffs
    }

    if len(filtered_test) != len(test_set) or len(filtered_val) != len(val_set):
        logger.warning(
            "Dropped %d test and %d val sequences whose FunFams are missing in training",
            len(test_set) - len(filtered_test),
            len(val_set) - len(filtered_val),
        )
        test_set = filtered_test
        val_set = filtered_val
        logger.warning(
            "Final test/val sizes: %d / %d (targets: %d / %d)",
            len(test_set), len(val_set), args.test_size, args.val_size,
        )

    write_fasta(args.output_dir / "train.fasta", train_set)
    write_fasta(args.output_dir / "val.fasta", val_set)
    write_fasta(args.output_dir / "test.fasta", test_set)

    def write_ids(path: Path, headers: dict[str, str]) -> None:
        with open(path, "w") as f:
            for header in headers:
                f.write(f"{extract_domain_id(header)}\n")

    write_ids(args.output_dir / "train_ids.txt", train_set)
    write_ids(args.output_dir / "val_ids.txt", val_set)
    write_ids(args.output_dir / "test_ids.txt", test_set)

    summary = {
        "params": {
            "test_size_target": args.test_size,
            "val_size_target": args.val_size,
            "min_seq_id": args.min_seq_id,
            "random_seed": args.random_seed,
        },
        "counts": {
            "deduplicated_sequences": len(dedup_sequences),
            "s30_clusters": len(clusters),
            "eligible_funfams": len(eligible_ffs),
            "train_sequences": len(train_set),
            "val_sequences": len(val_set),
            "test_sequences": len(test_set),
            "removed_train_by_identity": len(matching_train_ids),
        },
        "funfams": {
            "train": len({ff_mapping.get(extract_domain_id(h)) for h in train_set}),
            "val": len({ff_mapping.get(extract_domain_id(h)) for h in val_set}),
            "test": len({ff_mapping.get(extract_domain_id(h)) for h in test_set}),
        },
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    test_val_fasta.unlink(missing_ok=True)
    train_temp_fasta.unlink(missing_ok=True)

    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
