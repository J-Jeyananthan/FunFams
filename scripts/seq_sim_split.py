"""x seq identity per test vs train (MMseqs2, 80% bidirectional coverage)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_mmseqs_search(
    mmseqs: str,
    query_fasta: Path,
    target_fasta: Path,
    tmp_dir: Path,
    coverage: float,
    threads: int,
) -> Path:
    """Run MMseqs2 search with 80% bidirectional coverage. Returns path to hits TSV."""
    query_db = tmp_dir / "query_db"
    target_db = tmp_dir / "target_db"
    result_db = tmp_dir / "result_db"
    hits_tsv = tmp_dir / "hits.tsv"

    tmp_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([mmseqs, "createdb", str(query_fasta), str(query_db)], check=True)
    subprocess.run([mmseqs, "createdb", str(target_fasta), str(target_db)], check=True)
    subprocess.run(
        [
            mmseqs,
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp_dir),
            "--min-seq-id",
            "0",
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "-a",
            "--threads",
            str(threads),
        ],
        check=True,
    )
    subprocess.run(
        [
            mmseqs,
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(hits_tsv),
            "--format-output",
            "query,target,fident",
        ],
        check=True,
    )
    return hits_tsv


def load_max_identities(hits_tsv: Path) -> dict[str, tuple[float, str]]:
    """Return dict: query_id -> (max_identity_pct, best_target_id)."""
    result: dict[str, tuple[float, str]] = {}
    if not hits_tsv.exists():
        return result
    with hits_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, tid, fident = parts[0], parts[1], float(parts[2])
            pct = fident * 100.0
            if qid not in result or pct > result[qid][0]:
                result[qid] = (pct, tid)
    return result


def get_test_ids(test_fasta: Path) -> list[str]:
    """Extract sequence IDs from test FASTA in order."""
    ids: list[str] = []
    with test_fasta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Max seq identity per test vs train (MMseqs2, 80%% bidir cov).",
    )
    parser.add_argument("train_fasta", type=Path, help="Training FASTA.")
    parser.add_argument("test_fasta", type=Path, help="Test FASTA.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output TSV (default: stdout).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Bidirectional coverage threshold [0-1] (default: 0.8).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="MMseqs2 temp dir (default: <test_fasta>.mmseqs_tmp).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="MMseqs2 threads.",
    )
    parser.add_argument(
        "--mmseqs",
        default="mmseqs",
        help="MMseqs2 executable.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temp dir after run.",
    )

    args = parser.parse_args()
    train_fasta = args.train_fasta.resolve()
    test_fasta = args.test_fasta.resolve()
    tmp_dir = args.tmp_dir or (test_fasta.parent / f"{test_fasta.stem}.mmseqs_tmp")

    for p in [train_fasta, test_fasta]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    try:
        hits_tsv = run_mmseqs_search(
            mmseqs=args.mmseqs,
            query_fasta=test_fasta,
            target_fasta=train_fasta,
            tmp_dir=tmp_dir,
            coverage=args.coverage,
            threads=args.threads,
        )
        max_ids = load_max_identities(hits_tsv)
        test_ids = get_test_ids(test_fasta)
    finally:
        if tmp_dir.exists() and not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    lines = ["test_id\tmax_identity_pct\tbest_train_id"]
    for tid in test_ids:
        if tid in max_ids:
            pct, best_train = max_ids[tid]
            lines.append(f"{tid}\t{pct:.2f}\t{best_train}")
        else:
            lines.append(f"{tid}\tNA\t")

    out = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out, end="")


if __name__ == "__main__":
    main()