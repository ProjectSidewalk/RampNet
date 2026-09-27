import os, sys, json
W = "/gscratch/scrubbed/jfroehli/mv48"
sys.path.insert(0, os.path.join(W, "RampNet", "scripts", "model_comparison"))
os.chdir(os.path.join(W, "RampNet"))
import compare as C
b = os.path.join(W, "RampNet/benchmark/richmond_neighbourhood")
for spec in sys.argv[1:]:
    args = C.build_parser().parse_args([b, "--models", spec])
    records, verdicts, _ = C.load_bundle(b)
    prov, mid = C.parse_model_spec(spec)
    label, det = C.build_detector(prov, mid, records, args)
    sig = det.signature()
    cache = C.DetectionCache(os.path.join(W, "model_cache"))
    judged = list(verdicts)
    new = [p for p in records if p not in verdicts]
    hit = lambda ps: sum(cache.get(C.cache_key(label, sig, "richmond_neighbourhood", p)) is not None for p in ps)
    print(label, "judged cached", hit(judged), "/", len(judged), " new cached", hit(new), "/", len(new))
