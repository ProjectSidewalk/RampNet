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
SCORED = all(os.path.exists(os.path.join(DATA, p)) for p in
             ["yolo/sao_paulo_tiles.txt", "klone_sacct_D.txt"]
             + [f"rampnet_s{s}/sao_paulo.json" for s in sv.SEEDS_B])
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
    """Two readings, side by side: `statistics` is the pre-registered n=3 read on seeds
    1-3 of both arms and is NOT overwritten by the extension; `statistics_a2` is the same
    statistic over every scored Campaign B replicate (A2.3 item 2)."""
    result = sv.build()
    f1_a = [r["read"]["pooled"]["f1"] for r in result["campaign_a"].values()]
    f1_b_all = [result["campaign_b"][f"s{s}"]["read"]["pooled"]["f1"] for s in sv.SEEDS_B]
    f1_b_3 = [result["campaign_b"][f"s{s}"]["read"]["pooled"]["f1"] for s in sv.SEEDS_PREREG]
    for st, f1_b in ((result["statistics"], f1_b_3), (result["statistics_a2"], f1_b_all)):
        assert st["campaign_a_f1"] == f1_a and st["campaign_b_f1"] == f1_b
        assert st["n_A"] == 3 and st["n_B"] == len(f1_b)
        assert st["s_A"] == pytest.approx(statistics.stdev(f1_a))
        assert st["s_B"] == pytest.approx(statistics.stdev(f1_b))
        assert st["s_gap"] == pytest.approx(math.sqrt(st["s_A"] ** 2 + st["s_B"] ** 2))
        assert st["band_gap"] == sv.band_gap(st["s_gap"])
        assert st["band_b"] == sv.band_b(st["s_B"])
        assert st["welch_gap_of_means"]["diff"] == pytest.approx(st["gap_of_means"])
    assert result["statistics"]["seeds_b"] == [1, 2, 3]
    assert result["statistics_a2"]["seeds_b"] == list(sv.SEEDS_B)
    # s_A is not recomputed under Amendment 2 (A2.2: Campaign A is not extended).
    assert result["statistics_a2"]["s_A"] == result["statistics"]["s_A"]
    # The secondary best.pt legs are different numbers and are not in the statistic.
    f1_best = result["secondary"]["yolo_best_pt"]["f1"]
    assert all(v is not None for v in f1_best)
    assert f1_best != f1_a


# --------------------------------------------------------------------------- #
# Amendment 2: the Welch read and the klone restart record
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("df, expected", [(1, 12.7062), (2, 4.3027), (8, 2.3060),
                                          (10, 2.2281), (30, 2.0423), (100, 1.9840)])
def test_t_quantile_matches_the_table(df, expected):
    assert sv.t_quantile(0.975, df) == pytest.approx(expected, abs=5e-5)


def test_welch_is_the_satterthwaite_read_of_b_minus_a():
    a, b = [0.80, 0.81, 0.82], [0.83, 0.85, 0.84, 0.86]
    w = sv.welch(a, b)
    va, vb = statistics.variance(a) / 3, statistics.variance(b) / 4
    assert w["diff"] == pytest.approx(statistics.fmean(b) - statistics.fmean(a))
    assert w["se"] == pytest.approx(math.sqrt(va + vb))
    assert w["df"] == pytest.approx((va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 3))
    assert w["ci"][0] < w["diff"] < w["ci"][1]
    assert w["ci"][1] - w["diff"] == pytest.approx(w["t_crit"] * w["se"])
    assert sv.welch([0.8], b) is None          # one replicate has no variance


def test_sacct_D_counts_every_incarnation(tmp_path):
    p = tmp_path / "sacct.txt"
    p.write_text("JobID|JobName|State|Submit|Start|End|Elapsed|NodeList|Restarts|Partition|QOS\n"
                 "1|x|PREEMPTED|s|s|e|00:29:37|g1|0|ckpt-all|q\n"
                 "2|x|COMPLETED|s|s|e|03:50:11|g2|0|ckpt-all|q\n"
                 "1|x|PREEMPTED|s|s|e|00:11:16|g1|1|ckpt-all|q\n"
                 "1|x|COMPLETED|s|s|e|1-00:00:01|g1|2|ckpt-all|q\n", encoding="utf-8")
    jobs = sv.read_sacct(str(p))
    assert jobs[1]["incarnations"] == 3 and jobs[1]["restarts"] == 2
    assert jobs[1]["final_state"] == "COMPLETED"
    assert jobs[1]["elapsed_total_s"] == 29 * 60 + 37 + 11 * 60 + 16 + 86400 + 1
    assert jobs[2] == {"job": 2, "incarnations": 1, "restarts": 0, "elapsed_total_s": 13811,
                       "states": ["COMPLETED"], "final_state": "COMPLETED"}
    assert sv.read_sacct(str(tmp_path / "missing.txt")) == {}


@needs_scored
def test_every_campaign_b_replicate_completed_and_its_restarts_are_recorded():
    """A2.2: a requeued replicate is kept and its restart count travels beside it. The
    counts come from `sacct -D`; the .out log shows only the last incarnation."""
    result = sv.build()
    restarts = {s: result["campaign_b"][f"s{s}"]["klone_job"]["restarts"] for s in sv.SEEDS_B}
    assert restarts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 3, 6: 0, 7: 3, 8: 1, 9: 1}
    for s in sv.SEEDS_B:
        job = result["campaign_b"][f"s{s}"]["klone_job"]
        assert job["job"] == sv.CAMPAIGN_B_JOBS[s]
        assert job["final_state"] == "COMPLETED"
        assert job["incarnations"] == job["restarts"] + 1
        assert result["campaign_b"][f"s{s}"]["amendment_2"] == (s >= 4)


@needs_scored
def test_campaign_a_legs_are_the_picked_epochs():
    result = sv.build()
    for s in sv.SEEDS:
        r = result["campaign_a"][f"s{s}"]
        assert r["leg"] == f"y11x_tiles_s{s}_ep{r['epoch']}"
        assert r["epoch"] <= sv.MAX_EPOCH


@needs_scored
def test_rampnet_caches_are_the_single_pass_arm_at_the_published_floor():
    for s in sv.SEEDS_B:
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
    with open(os.path.join(DATA, "env_amend2_2026-09-17.txt"), encoding="utf-8") as fh:
        env_a2 = fh.read()
    for s in sv.SEEDS_A:
        assert f"{sv.primary_leg(s)}.pt" in env
    for s in sv.SEEDS_PREREG:
        assert f"rampnet_s{s}_best.pth" in env
    for s in sv.SEEDS_B:
        if s not in sv.SEEDS_PREREG:
            assert f"rampnet_s{s}_best.pth" in env_a2


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


@needs_scored
def test_the_two_readings_are_pinned_side_by_side():
    """The pre-registered n=3 read stays as scored; the Amendment 2 read sits beside it.
    Both are re-derived by --check; this pins them against a silent regeneration."""
    with open(sv.OUT_JSON, encoding="utf-8") as fh:
        art = json.load(fh)
    n3, n9 = art["statistics"], art["statistics_a2"]
    assert n3["seeds_b"] == [1, 2, 3] and n9["seeds_b"] == list(range(1, 10))
    assert n3["campaign_b_f1"] == n9["campaign_b_f1"][:3]
    assert (n3["s_B"], n3["s_gap"], n3["band_gap"]) == (0.01114, 0.01142, "ambiguous")
    assert (n9["s_B"], n9["s_gap"], n9["band_gap"]) == (0.00941, 0.00974, "real")
    assert n9["s_A"] == n3["s_A"] == 0.00252
    w = n9["welch_gap_of_means"]
    assert w["ci"] == [0.00814, 0.02355] and round(w["df"], 1) == 10.0
    # The n=3 CI includes zero, the n=9 one does not -- and neither includes 0.039.
    assert n3["welch_gap_of_means"]["ci"][0] < 0 < n3["welch_gap_of_means"]["ci"][1]
    assert 0 < w["ci"][0] and w["ci"][1] < n9["gap_published"]
