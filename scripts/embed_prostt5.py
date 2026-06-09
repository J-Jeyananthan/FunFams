#!/usr/bin/env python3
"""
Generate ProstT5 embeddings for protein sequences in FASTA format and store in LMDB.

Streams FASTA input (no full-file load), embeds using the ProstT5 encoder on a single GPU,
and writes mean-pooled float16 embeddings to an LMDB database. Resume-safe: reruns skip
sequences already present in the LMDB.

Key behaviors:
- Streams FASTA (no full-file load)
- Single GPU (cuda:0 if available)
- Single LMDB environment
- Resume: if key exists and overwrite==0, skip embedding
- Writes embeddings as NumPy .npy blobs
- Optionally maintains a keys index file; can rebuild it from LMDB

Example (resume-safe):
  CUDA_VISIBLE_DEVICES=0 python embed_lmdb_single_resume.py \
    -i sequences.fasta -o embeddings.lmdb \
    --per_protein 1 --half 1 --storage_dtype float16 \
    --map_size_gb 256 --commit_every 5000 \
    --overwrite 0 --resume 1
"""

import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import lmdb
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prostt5_lmdb_single_resume")


# ----------------------------
# FASTA streaming
# ----------------------------
def iter_fasta_records(
    fasta_path: Path,
    split_char: str,
    id_field: int,
    is_3Di: bool,
) -> Generator[Tuple[str, str], None, None]:
    """
    Stream FASTA records as (protein_id, sequence).
    - Extracts ID using split_char and id_field
    - Cleans ID: '/' and '.' -> '_'
    - Removes '-' and whitespace; lowercases if is_3Di
    """
    cur_id: Optional[str] = None
    cur_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            if not line:
                continue

            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(cur_seq_parts)

                header = line[1:].strip()
                parts = header.split(split_char)
                raw_id = parts[id_field] if id_field < len(parts) else header
                cur_id = raw_id.replace("/", "_").replace(".", "_")
                cur_seq_parts = []
            else:
                s = "".join(line.split()).replace("-", "")
                if is_3Di:
                    s = s.lower()
                cur_seq_parts.append(s)

    if cur_id is not None:
        yield cur_id, "".join(cur_seq_parts)


# ----------------------------
# Model / device
# ----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_T5_model(model_dir: str, device: torch.device):
    logger.info(f"Loading T5 from: {model_dir}")
    model = T5EncoderModel.from_pretrained(model_dir).to(device).eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab


# ----------------------------
# LMDB helpers
# ----------------------------
def npy_dumps(array: np.ndarray) -> bytes:
    """Serialize numpy array into .npy bytes (dtype+shape included)."""
    bio = io.BytesIO()
    np.save(bio, array, allow_pickle=False)
    return bio.getvalue()


def open_lmdb_env(path: Path, map_size_gb: int, subdir: bool) -> lmdb.Environment:
    if subdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(path),
        map_size=int(map_size_gb) * (1024**3),
        subdir=subdir,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=2,
    )
    return env


def grow_map(env: lmdb.Environment, min_grow_bytes: int = 1 * (1024**3)) -> None:
    info = env.info()
    cur = int(info.get("map_size", 0))
    new = max(cur * 2, cur + min_grow_bytes)
    logger.warning(f"LMDB MapFullError: growing map from {cur/1024**3:.1f} GB -> {new/1024**3:.1f} GB")
    env.set_mapsize(new)


def rebuild_keys_file(env: lmdb.Environment, emb_db: lmdb._Database, keys_path: Path) -> None:
    """
    Rebuild a complete keys file by scanning the LMDB.
    WARNING: For huge DBs this can take a while; do it once at the end.
    """
    logger.info(f"Rebuilding keys file from LMDB into: {keys_path}")
    tmp = keys_path.with_suffix(keys_path.suffix + ".tmp")
    n = 0
    with env.begin(db=emb_db, write=False) as txn, open(tmp, "w") as out:
        with txn.cursor() as cur:
            for k, _v in cur:
                # Skip internal/meta-like keys if any were ever stored here
                if k.startswith(b"__"):
                    continue
                out.write(k.decode("utf-8") + "\n")
                n += 1
                if n % 1_000_000 == 0:
                    logger.info(f"Rebuilt keys: {n:,}")
    tmp.replace(keys_path)
    logger.info(f"Keys rebuild complete: {n:,} keys")


# ----------------------------
# Main embedding loop (resume-aware)
# ----------------------------
def embed_fasta_to_lmdb(
    fasta_path: Path,
    out_lmdb: Path,
    model_dir: str,
    split_char: str,
    id_field: int,
    per_protein: bool,
    half_precision: bool,
    is_3Di: bool,
    storage_dtype: str,
    map_size_gb: int,
    commit_every: int,
    subdir: bool,
    overwrite: bool,
    resume: bool,
    write_keys_txt: bool,
    keys_mode: str,
    rebuild_keys: bool,
    max_residues: int,
    max_seq_len: int,
    max_batch: int,
) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"

    model, vocab = get_T5_model(model_dir, device)
    if half_precision:
        model = model.half()
        logger.info("Using model in half-precision")

    if storage_dtype not in ("float16", "float32"):
        raise ValueError("--storage_dtype must be float16 or float32")

    logger.info(f"Writing LMDB to: {out_lmdb}")
    env = open_lmdb_env(out_lmdb, map_size_gb=map_size_gb, subdir=subdir)
    emb_db = env.open_db(b"embeddings")
    meta_db = env.open_db(b"meta")

    # write meta (idempotent overwrite)
    meta = {
        "model": model_dir,
        "per_protein": bool(per_protein),
        "half_precision_model": bool(half_precision),
        "is_3Di": bool(is_3Di),
        "storage_dtype": storage_dtype,
        "created_unix": int(time.time()),
        "prefix": prefix,
        "resume": bool(resume),
        "overwrite": bool(overwrite),
    }
    with env.begin(write=True, db=meta_db) as txn:
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    # keys file handling
    keys_path = out_lmdb.with_suffix(out_lmdb.suffix + ".keys.txt")
    keys_fh = None
    if write_keys_txt and not rebuild_keys:
        if keys_mode not in ("append", "truncate"):
            raise ValueError("--keys_mode must be append or truncate")

        mode = "a" if keys_mode == "append" else "w"
        keys_fh = open(keys_path, mode)
        logger.info(f"Keys index file: {keys_path} (mode={mode})")

    start = time.time()
    wrote = 0
    skipped_existing = 0
    failed = 0
    seen = 0  # FASTA records seen

    # Batch holds only items we will embed (not already present)
    batch = []

    # Write transaction (we'll also do existence checks against it; safe)
    txn = env.begin(write=True, db=emb_db)

    def commit_txn():
        nonlocal txn
        txn.commit()
        txn = env.begin(write=True, db=emb_db)

    def key_exists(key: bytes) -> bool:
        # Resume is only meaningful when overwrite is False
        if not resume or overwrite:
            return False
        return txn.get(key) is not None

    try:
        for pid, seq_raw in tqdm(iter_fasta_records(fasta_path, split_char, id_field, is_3Di),
                                 desc="Embedding", unit="seq"):
            seen += 1
            key = pid.encode("utf-8")

            # RESUME SKIP: if already in LMDB, skip before any GPU work
            if key_exists(key):
                skipped_existing += 1
                continue

            # Replace non-standard amino acids
            seq = seq_raw.replace("U", "X").replace("Z", "X").replace("O", "X")
            s_len = len(seq)

            # Token string: prefix + spaced characters
            seq_tok = prefix + " " + " ".join(list(seq))
            batch.append((pid, key, seq_tok, s_len))

            n_res_batch = sum(x[3] for x in batch)
            if len(batch) >= max_batch or n_res_batch >= max_residues or s_len > max_seq_len:
                wrote, failed = _flush_batch(
                    batch=batch,
                    model=model,
                    vocab=vocab,
                    device=device,
                    per_protein=per_protein,
                    storage_dtype=storage_dtype,
                    overwrite=overwrite,
                    txn_ref=lambda: txn,
                    set_txn=lambda t: _set_txn_locals(t, locals_dict={"txn": None}),  # not used
                    env=env,
                    emb_db=emb_db,
                    keys_fh=keys_fh,
                    wrote=wrote,
                    failed=failed,
                    example_log=(wrote == 0),
                )
                batch = []

                if commit_every > 0 and wrote > 0 and wrote % commit_every == 0:
                    commit_txn()

        # Flush leftover
        if batch:
            wrote, failed = _flush_batch(
                batch=batch,
                model=model,
                vocab=vocab,
                device=device,
                per_protein=per_protein,
                storage_dtype=storage_dtype,
                overwrite=overwrite,
                txn_ref=lambda: txn,
                set_txn=None,
                env=env,
                emb_db=emb_db,
                keys_fh=keys_fh,
                wrote=wrote,
                failed=failed,
                example_log=(wrote == 0),
            )
            batch = []

        txn.commit()

    except Exception:
        try:
            txn.abort()
        except Exception:
            pass
        raise
    finally:
        if keys_fh is not None:
            keys_fh.flush()
            keys_fh.close()
        env.sync()

        # Optionally rebuild keys file from LMDB at end (safe after resumes)
        if write_keys_txt and rebuild_keys:
            try:
                rebuild_keys_file(env, emb_db, keys_path)
            except Exception as e:
                logger.warning(f"Failed to rebuild keys file: {e}")

        env.close()

    dt = time.time() - start
    logger.info(f"FASTA records seen: {seen}")
    logger.info(f"Wrote embeddings: {wrote}")
    if skipped_existing:
        logger.info(f"Skipped existing (resume): {skipped_existing}")
    if failed:
        logger.info(f"Failed (runtime errors): {failed}")
    if wrote:
        logger.info(f"Time: {dt:.2f}s | {dt/wrote:.5f} s/protein (written only)")


def _flush_batch(
    batch,
    model,
    vocab,
    device,
    per_protein,
    storage_dtype,
    overwrite,
    txn_ref,
    set_txn,
    env,
    emb_db,
    keys_fh,
    wrote,
    failed,
    example_log: bool,
):
    """
    Embed + write one batch. Returns updated (wrote, failed).
    batch entries: (pid_str, key_bytes, seq_tok_str, seq_len_int)
    """
    pdb_ids, keys, seqs, seq_lens = zip(*batch)

    token_encoding = vocab.batch_encode_plus(
        seqs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    try:
        with torch.no_grad():
            out = model(
                token_encoding.input_ids,
                attention_mask=token_encoding.attention_mask,
            )
    except RuntimeError as e:
        logger.warning(f"RuntimeError during embedding batch (last id {pdb_ids[-1]}): {e}")
        failed += len(pdb_ids)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return wrote, failed

    txn = txn_ref()

    for i, identifier in enumerate(pdb_ids):
        L = seq_lens[i]
        emb = out.last_hidden_state[i, 1 : L + 1]  # account for prefix token
        if per_protein:
            emb = emb.mean(dim=0)  # (d,)

        arr = emb.detach().cpu().numpy().squeeze()
        arr = arr.astype(np.float16 if storage_dtype == "float16" else np.float32, copy=False)

        val = npy_dumps(arr)
        key = keys[i]

        while True:
            try:
                txn.put(key, val, overwrite=overwrite)
                break
            except lmdb.MapFullError:
                txn.abort()
                grow_map(env)
                txn = env.begin(write=True, db=emb_db)

        wrote += 1
        if keys_fh is not None:
            keys_fh.write(identifier + "\n")

        if example_log and wrote == 1:
            logger.info(f"Example embedded {identifier} -> dtype {arr.dtype}, shape {arr.shape}")

    return wrote, failed


# Helper for mypy/linters; not actually used
def _set_txn_locals(_t, locals_dict):
    locals_dict["txn"] = _t


# ----------------------------
# CLI
# ----------------------------
def create_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate ProstT5 embeddings and write to a single LMDB (with resume).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("-i", "--input", required=True, type=str, help="Input FASTA file")
    p.add_argument("-o", "--output", required=True, type=str,
                   help="Output LMDB path (directory by default; file if --subdir 0)")

    p.add_argument("--model", type=str, default="Rostlab/ProstT5", help="Model dir or HF id")
    p.add_argument("--split_char", type=str, default="!", help="Split character for FASTA header")
    p.add_argument("--id", type=int, default=0, help="Field index after split for ID extraction")

    p.add_argument("--per_protein", type=int, default=1, choices=[0, 1], help="Mean-pooled per-protein if 1")
    p.add_argument("--half", type=int, default=1, choices=[0, 1], help="Use half-precision model weights if 1")
    p.add_argument("--is_3Di", type=int, default=0, choices=[0, 1], help="Input sequences are 3Di if 1")

    p.add_argument("--storage_dtype", type=str, default="float16", choices=["float16", "float32"],
                   help="Dtype to store embeddings in LMDB")
    p.add_argument("--map_size_gb", type=int, default=64, help="Initial LMDB map size (auto-grows)")
    p.add_argument("--commit_every", type=int, default=5000, help="Commit LMDB every N written embeddings")

    p.add_argument("--subdir", type=int, default=1, choices=[0, 1],
                   help="If 1, LMDB is a directory env; if 0, single-file env")

    # Resume/overwrite controls
    p.add_argument("--resume", type=int, default=1, choices=[0, 1],
                   help="If 1 and overwrite=0, skip IDs already present in LMDB")
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1],
                   help="If 1, overwrite keys if they exist; if 0, keep existing and (optionally) resume-skip")

    # Keys index controls
    p.add_argument("--write_keys_txt", type=int, default=1, choices=[0, 1],
                   help="Write a .keys.txt file for Dataset indexing")
    p.add_argument("--keys_mode", type=str, default="append", choices=["append", "truncate"],
                   help="If writing keys file: append to existing or truncate and rewrite (not resume-safe)")
    p.add_argument("--rebuild_keys", type=int, default=0, choices=[0, 1],
                   help="If 1, rebuild keys file by scanning LMDB at end (slow for huge DBs, but complete)")

    # batching knobs
    p.add_argument("--max_residues", type=int, default=4000, help="Max residues per batch")
    p.add_argument("--max_seq_len", type=int, default=1000, help="Max sequence length")
    p.add_argument("--max_batch", type=int, default=100, help="Max sequences per batch")

    return p


def main():
    args = create_arg_parser().parse_args()

    fasta_path = Path(args.input)
    out_lmdb = Path(args.output)

    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    embed_fasta_to_lmdb(
        fasta_path=fasta_path,
        out_lmdb=out_lmdb,
        model_dir=args.model,
        split_char=args.split_char,
        id_field=args.id,
        per_protein=bool(args.per_protein),
        half_precision=bool(args.half),
        is_3Di=bool(args.is_3Di),
        storage_dtype=args.storage_dtype,
        map_size_gb=args.map_size_gb,
        commit_every=args.commit_every,
        subdir=bool(args.subdir),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        write_keys_txt=bool(args.write_keys_txt),
        keys_mode=args.keys_mode,
        rebuild_keys=bool(args.rebuild_keys),
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch,
    )


if __name__ == "__main__":
    main()
