"""Build per-leg shard bundles for the #48 klone offload.

Each shard is benchmark/richmond_neighbourhood (the directory NAME is the cache key's
city, so it must not change) holding the 124 judged records plus 1/N of the new panos,
a bundle.json copy, a ../richmond symlink for verdicts_from, and pano symlinks.
Usage: python make_shards.py <W> leg:N [leg:N ...]
"""
import json, os, sys
W = sys.argv[1]
repo = os.path.join(W, "RampNet")
src = os.path.join(repo, "benchmark", "richmond_neighbourhood")
new = [l.strip() for l in open(os.path.join(W, "new_panos.txt")) if l.strip()]
judged = set(json.load(open(os.path.join(repo, "benchmark", "richmond", "verdicts.json")))["panos"])
recs = {}
for line in open(os.path.join(src, "records.jsonl"), encoding="utf-8"):
    if line.strip():
        recs[json.loads(line)["pano"]["panorama_id"]] = line.rstrip("\n")
assert len(new) == 1307 and all(p in recs for p in new) and judged <= set(recs), "pano lists drifted"
assert not (judged & set(new))
for spec in sys.argv[2:]:
    leg, n = spec.split(":"); n = int(n)
    for k in range(n):
        d = os.path.join(W, "shards", leg, f"s{k}", "benchmark")
        b = os.path.join(d, "richmond_neighbourhood")
        os.makedirs(os.path.join(b, "panos"), exist_ok=True)
        if not os.path.lexists(os.path.join(d, "richmond")):
            os.symlink(os.path.join(repo, "benchmark", "richmond"), os.path.join(d, "richmond"))
        mine = [p for p in recs if p in judged] + new[k::n]
        with open(os.path.join(b, "records.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(recs[p] for p in mine) + "\n")
        with open(os.path.join(src, "bundle.json"), encoding="utf-8") as fi, open(os.path.join(b, "bundle.json"), "w") as fo:
            fo.write(fi.read())
        for p in mine:
            dst = os.path.join(b, "panos", p + ".jpg")
            s = os.path.join(W, "panos", p + ".jpg")
            assert os.path.exists(s), s
            if not os.path.lexists(dst):
                os.symlink(s, dst)
        print(leg, k, len(mine))
