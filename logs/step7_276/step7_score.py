"""Step 7 scratch: Flash expert+conf vs zs+conf on the open pool, original run + 2 replicates."""
import sys, json
sys.path.insert(0, r"D:\Git\RampNet\scripts\analysis"); sys.path.insert(0, r"D:\Git\RampNet")
import numpy as np
import prompt_276_heldout as H
C, D = H.C, H.D
rsq = H.radius_sq_for()
caches = {"orig": r"D:\Git\RampNet\.model_cache", "rep1": r"D:\Git\RampNet\.model_cache_rep1", "rep2": r"D:\Git\RampNet\.model_cache_rep2"}
gts = {s: H.load_split_gts(s) for s in H.OPEN_POOL}
dev_records, _, _ = C.load_bundle(r"D:\Git\RampNet\benchmark\sao_paulo")
out = {}
for run, cdir in caches.items():
    cache = C.DetectionCache(cdir, enabled=True)
    args = H._compare_args(cdir); args.gemini_confidence = True; args.gemini_thinking_level = None
    cnt = {}
    for arm, definition in (("zs", None), ("expert", "expert")):
        args.definition = definition
        label, det = D.build_detector("gemini", "gemini-3.8-flash", dev_records, args)
        sig = det.signature(); cnt[arm] = {}
        for s in H.OPEN_POOL:
            city, g = gts[s]
            pts, missing = H.cached_points(cache, label, sig, city, list(g))
            assert not missing, (run, arm, s, len(missing))
            cnt[arm][s] = H.pano_counts(pts, g, rsq, 0.0)
    out[run] = cnt
    f = {a: {s: H.f1_from_counts(*cnt[a][s].sum(0)) for s in H.OPEN_POOL} for a in cnt}
    d, lo, hi, p = H.paired_bootstrap(cnt["expert"], cnt["zs"], np.random.default_rng(H.SEED))
    print(run, {a: {s: round(v, 3) for s, v in f[a].items()} for a in f},
          "macro zs %.3f expert %.3f" % (np.mean(list(f["zs"].values())), np.mean(list(f["expert"].values()))),
          "delta %.4f [%.4f, %.4f] p %s" % (d, lo, hi, p))
runs = list(out)
for a in ("zs", "expert"):
    m = [np.mean([H.f1_from_counts(*out[r][a][s].sum(0)) for s in H.OPEN_POOL]) for r in runs]
    print(a, "macro F1 across runs", [round(x, 3) for x in m], "range %.3f sd %.3f" % (max(m) - min(m), np.std(m, ddof=1)))
