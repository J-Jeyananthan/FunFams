python - <<'EOF'
import lmdb
from pathlib import Path

mapping_path = Path("/SAN/orengolab/functional-families/janu/data/duplicates_removed-funfams-4.3-c123-mapping.txt")
lmdb_path = "/SAN/orengolab/functional-families/janu/contrasted-ff/funfams-4.3-c123.lmdb"

env = lmdb.open(lmdb_path, readonly=True, lock=False)

missing = []
bad = 0

with env.begin() as txn:
    print("LMDB entries:", txn.stat()["entries"])

    with mapping_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                bad += 1
                continue

            header_raw = parts[0]
            label = parts[1]
            header_key = header_raw.replace("/", "_")

            if txn.get(header_key.encode("utf-8")) is None:
                missing.append((header_raw, header_key, label))

            if lineno % 1_000_000 == 0:
                print(f"Read {lineno:,} lines... missing so far: {len(missing)}")

print(f"\nBad lines skipped: {bad:,}")
print(f"Missing found: {len(missing)}")

out = Path("missing_4_from_lmdb.txt")
out.write_text("".join(f"{raw}\t{key}\t{label}\n" for raw, key, label in missing), encoding="utf-8")
print("Wrote:", out.resolve())

for raw, key, label in missing:
    print("MISSING:", raw, "->", key, "\t", label)
EOF
