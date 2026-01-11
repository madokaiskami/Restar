import argparse
import gzip
import json
import os


def iter_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    from restar.utils import stable_id

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="", help="MANIFEST.json that lists eval jsonl paths")
    ap.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="Explicit list of jsonl paths to include",
    )
    ap.add_argument("--dev_jsonl", default="", help="Dev set jsonl to include")
    ap.add_argument("--out", required=True, help="Output blocklist path")
    args = ap.parse_args()

    files = []
    if args.manifest and os.path.exists(args.manifest):
        try:
            m = json.load(open(args.manifest, "r", encoding="utf-8"))
            files.extend(m.get("files", []))
        except Exception:
            pass
    files.extend(args.paths or [])

    S = set()
    for p in files:
        for row in iter_jsonl(p):
            rid = stable_id(row.get("text", ""), row.get("asin", ""))
            S.add(rid)
    if args.dev_jsonl and os.path.exists(args.dev_jsonl):
        for row in iter_jsonl(args.dev_jsonl):
            rid = stable_id(row.get("text", ""), row.get("asin", ""))
            S.add(rid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for h in sorted(S):
            f.write(h + "\n")
    print(f"Blocklist size={len(S)} -> {args.out}")


if __name__ == "__main__":
    main()
