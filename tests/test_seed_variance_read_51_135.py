"""The pre-registered seed-variance read (#51, #135), pinned.

CPU only, committed fixtures only. What is load-bearing here:

* the epoch pick reads the column the pre-registration names (``metrics/mAP50-95(B)``),
  and the committed ``results.csv`` files make s2 the case where the fitness blend
  would have chosen a different epoch;
* the Amendment 1 bands are disjoint and each endpoint belongs to exactly one row;
* ``s_gap`` is the root-sum-square of the two campaign SDs, never ``s_A`` alone;
* the secondary reads cannot leak into the decision statistic;
* once the scored inputs are committed, the artifact reproduces from them and every
  primary threshold was selected on ``sao_paulo`` and no pooled split.
"""
import json
import math
import os
import statistics
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

import seed_variance_read_51_135 as sv  # noqa: E402

DATA = os.path.join(ROOT, "docs", "data", "seed_variance_51_135")
SCORED = all(os.path.exists(os.path.join(DATA, p)) for p in (
    "yolo/sao_paulo_tiles.txt", "rampnet_s1/sao_paulo.json",
    "rampnet_s2/sao_paulo.json", "rampnet_s3/sao_paulo.json"))
needs_scored = pytest.mark.skipif(not SCORED, reason="scored inputs not committed yet")


# --------------------------------------------------------------------------- #
# the epoch pick
# --------------------------------------------------------------------------- #
def test_pick_epoch_reads_the_named_column_and_gives_44_44_42():
    picks = {s: sv.pick_epoch(os.path.join(DATA, f"y11x_tiles_s{s}", "results.csv"))[0]
             for s in sv.SEEDS}
    assert picks == {1: 44, 2: 44, 3: 42}


def test_s2_is_where_the_blend_and_map5095_disagree():
    """The pre-registration names the column because they do not always peak
    together. s2 is the live example: blend -> ep41, mAP50-95 -> ep44, by 0.00006."""
    csv = os.path.join(DATA, "y11x_tiles_s2", "results.csv")
    assert sv.fitness_epoch(csv) == 41
    assert sv.pick_epoch(csv)[0] == 44
    for s in (1, 3):
        csv = os.path.join(DATA, f"y11x_tiles_s{s}", "results.csv")
        assert sv.fitness_epoch(csv) == sv.pick_epoch(csv)[0]


def test_pick_epoch_respects_the_ceiling():
    csv = os.path.join(DATA, "y11x_tiles_s1", "results.csv")
    ep60, v60 = sv.pick_epoch(csv, max_epoch=60)
    ep44, v44 = sv.pick_epoch(csv, max_epoch=44)
    assert ep44 <= 44 < ep60 and v60 > v44


def test_pick_epoch_ties_go_to_the_earlier_epoch(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.4\n2,0.6,0.4\n3,0.7,0.3\n")
    assert sv.pick_epoch(str(p)) == (1, 0.4)


def test_every_replicate_trained_the_same_config_except_the_seed():
    """args.yaml differs only in seed and the run-specific paths."""
    import re
    cfgs = {}
    for s in sv.SEEDS:
        with open(os.path.join(DATA, f"y11x_tiles_s{s}", "args.yaml"), encoding="utf-8") as fh:
            cfgs[s] = dict(line.split(":", 1) for line in fh.read().splitlines() if ":" in line)
    keys = set(cfgs[1])
    assert all(set(c) == keys for c in cfgs.values())
    varying = {k for k in keys if len({cfgs[s][k].strip() for s in sv.SEEDS}) > 1}
    for k in varying - {"seed"}:
        assert all(re.search(r"y11x_tiles_s\d", cfgs[s][k]) for s in sv.SEEDS), k
    assert {cfgs[s]["seed"].strip() for s in sv.SEEDS} == {"1", "2", "3"}
    assert cfgs[1]["epochs"].strip() == "60" and cfgs[1]["imgsz"].strip() == "1024"


# --------------------------------------------------------------------------- #
# the bands and the arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s, band", [
    (0.0, "real"), (0.00999, "real"),
    (0.010, "ambiguous"), (0.01999, "ambiguous"),
    (0.020, "indistinguishable"), (0.5, "indistinguishable"),
])
def test_bands_are_disjoint_with_each_endpoint_in_one_row(s, band):
    assert sv.band_gap(s) == band


def test_band_b_reads_against_the_paired_mde():
    assert sv.PAIRED_MDE_135 == 0.0063
    assert sv.band_b(0.0062) == "below_paired_mde"
    assert sv.band_b(0.0063) == "dominates_paired_mde"
    assert sv.band_b(None) is None


def test_sd_is_the_sample_sd_and_needs_the_replicates():
    assert sv.sd([0.80, 0.81, 0.82]) == pytest.approx(statistics.stdev([0.80, 0.81, 0.82]))
    assert sv.sd([0.8, None, None]) is None


def test_s_gap_is_root_sum_square_not_s_a_alone():
    # A1.1's argument: an s_A that reads "real" on its own stops reading that way once
    # the RampNet side's noise is added in. 0.009 alone is inside the real band; with
    # s_B = 0.009 the gap's sigma is 0.0127 and the band is ambiguous.
    s_a, s_b = 0.009, 0.009
    s_gap = math.sqrt(s_a ** 2 + s_b ** 2)
    assert s_gap == pytest.approx(0.01273, abs=1e-5)
    assert sv.band_gap(s_a) == "real"
    assert sv.band_gap(s_gap) == "ambiguous"
    assert 0.039 / s_gap == pytest.approx(3.06, abs=0.01)


def test_published_reference_is_the_parity_artifact():
    ref = sv.published_reference()
    assert ref["gap"] == pytest.approx(0.039, abs=0.0005)
    assert ref["rampnet_threshold"] == 0.30 and ref["yolo_threshold"] == 0.10


def test_leg_names_follow_the_checkpoint_stems():
    assert sv.primary_leg(2) == "y11x_tiles_s2_ep44"
    assert sv.secondary_leg(2) == "y11x_tiles_s2_best"


# --------------------------------------------------------------------------- #
# once the scored inputs are committed
# --------------------------------------------------------------------------- #
@needs_scored
def test_committed_artifact_is_current():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analysis",
                                                     "seed_variance_read_51_135.py"),
                        "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


@needs_scored
def test_primary_thresholds_were_selected_on_the_dev_split_only():
    result = sv.build()
    for camp in ("campaign_a", "campaign_b"):
        for k, r in result[camp].items():
            assert r["read"] is not None, (camp, k)
            dev = r["read"]["per_split"][sv.DEFAULT_DEV]
            thr = r["read"]["selected_threshold"]
            # The selected threshold maximises F1 on the dev split ...
            sweep = (sv.collect_yolo_seed()[r["leg"]] if camp == "campaign_a"
                     else sv.collect_rampnet_seed(int(k[1:]), result["grid_rampnet"]))
            assert dev["f1"] == max(c["f1"] for c in sweep[sv.DEFAULT_DEV].values())
            # ... and is the same threshold on every pooled split (uniform, not per-split).
            assert all(s in r["read"]["per_split"] for s in sv.POOLED_SPLITS)
            assert r["read"]["pooled"]["threshold"] == thr


@needs_scored
def test_primary_pooled_f1_is_the_macro_mean_over_the_seven_splits():
    result = sv.build()
    for camp in ("campaign_a", "campaign_b"):
        for r in result[camp].values():
            per = r["read"]["per_split"]
            macro = sum(per[s]["f1"] for s in sv.POOLED_SPLITS) / len(sv.POOLED_SPLITS)
            assert r["read"]["pooled"]["f1"] == pytest.approx(macro)
            assert r["read"]["pooled"]["n_splits"] == 7


@needs_scored
def test_statistics_are_computed_from_the_primary_reads_only():
    result = sv.build()
    st = result["statistics"]
    f1_a = [r["read"]["pooled"]["f1"] for r in result["campaign_a"].values()]
    f1_b = [r["read"]["pooled"]["f1"] for r in result["campaign_b"].values()]
    assert st["campaign_a_f1"] == f1_a and st["campaign_b_f1"] == f1_b
    assert st["s_A"] == pytest.approx(statistics.stdev(f1_a))
    assert st["s_B"] == pytest.approx(statistics.stdev(f1_b))
    assert st["s_gap"] == pytest.approx(math.sqrt(st["s_A"] ** 2 + st["s_B"] ** 2))
    assert st["band_gap"] == sv.band_gap(st["s_gap"])
    assert st["band_b"] == sv.band_b(st["s_B"])
    # The secondary best.pt legs are different numbers and are not in the statistic.
    f1_best = result["secondary"]["yolo_best_pt"]["f1"]
    assert all(v is not None for v in f1_best)
    assert f1_best != f1_a


@needs_scored
def test_campaign_a_legs_are_the_picked_epochs():
    result = sv.build()
    for s in sv.SEEDS:
        r = result["campaign_a"][f"s{s}"]
        assert r["leg"] == f"y11x_tiles_s{s}_ep{r['epoch']}"
        assert r["epoch"] <= sv.MAX_EPOCH


@needs_scored
def test_rampnet_caches_are_the_single_pass_arm_at_the_published_floor():
    for s in sv.SEEDS:
        for split in sv.SPLITS_AS_RUN:
            with open(os.path.join(DATA, f"rampnet_s{s}", f"{split}.json"), encoding="utf-8") as fh:
                meta = json.load(fh)["meta"]
            assert meta["model"] == f"rampnet_s{s}"
            assert meta["score_floor"] == 0.05 and meta["min_distance"] == 10
            assert not meta.get("tta", False)


@needs_scored
def test_every_yolo_report_carries_all_six_legs():
    for split in sv.SPLITS_AS_RUN:
        path = os.path.join(DATA, "yolo", f"{split}_tiles.txt")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for s in sv.SEEDS:
            assert f"[{sv.primary_leg(s)}] threshold sweep" in text, (split, s)
            assert f"[{sv.secondary_leg(s)}] threshold sweep" in text, (split, s)


@needs_scored
def test_env_records_the_checkpoint_hashes_the_run_used():
    with open(os.path.join(DATA, "env.txt"), encoding="utf-8") as fh:
        env = fh.read()
    for s in sv.SEEDS:
        assert f"{sv.primary_leg(s)}.pt" in env and f"rampnet_s{s}_best.pth" in env


# --------------------------------------------------------------------------- #
# the 2026-09-17 correction: the primary legs were re-scored on the right checkpoints
# --------------------------------------------------------------------------- #
MISLABELLED = os.path.join(DATA, "yolo_mislabelled_ep45_45_43")


@needs_scored
def test_corrected_sweeps_differ_from_the_mislabelled_ones_only_on_the_ep_legs():
    """The 09-15 files scored epoch44/44/42.pt, which are results.csv epochs 45/45/43.
    The re-score swapped the checkpoints and nothing else, so the three best.pt control
    legs must reproduce cell-for-cell and the three _ep legs must not."""
    def cells(yolo_dir):  # collect_yolo_seed expects <data>/yolo/; the archive is flat
        out = {}
        for split in sv.SPLITS_AS_RUN:
            for model, sweep in sv.parse_sweeps(os.path.join(yolo_dir, f"{split}_tiles.txt")).items():
                out.setdefault(model, {})[split] = sweep
        return out
    old = cells(MISLABELLED)
    new = cells(os.path.join(DATA, "yolo"))
    assert set(old) == set(new)
    for leg in new:
        same = all(new[leg][sp][t] == old[leg][sp].get(t)
                   for sp in new[leg] for t in new[leg][sp])
        if leg.endswith("_best"):
            assert same, f"{leg}: the control leg should reproduce exactly"
        else:
            assert not same, f"{leg}: the corrected leg scored identically to the wrong one"


@needs_scored
def test_the_corrected_primary_read_is_pinned():
    """Guards against the mislabelled files ever being copied back over the read."""
    with open(sv.OUT_JSON, encoding="utf-8") as fh:
        stats = json.load(fh)["statistics"]
    assert stats["campaign_a_f1"] == [0.80614, 0.80486, 0.80971]
    assert stats["campaign_a_f1"] != [0.81286, 0.80871, 0.80900]  # the 09-15 numbers
