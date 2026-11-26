"""
Step 6 -- collect every result into tables and figures.

  python make_paper_numbers.py --cache data_cache --runs runs --cost model_cost.json --paper results

Writes (into the --paper folder)
  results_macros.tex                  every reported number, as \resdef{key}{value} (for LaTeX)
  results_summary.md                  the same numbers in a readable list
  figures/fig_compression.pdf/.png    F1 / AUROC versus DBBD block size
  figures/fig_confusion.pdf/.png      health confusion matrix at b = 2
  figures/fish_b1/b2/b4.png           one test fish at b = 1, 2, 4
  figures/fig_dbbd_levels.png         the same fish as one labelled strip
"""
import argparse, glob, json, os

import numpy as np

HEALTH_KEYS = {"Healthy": "healthy", "Bacterial gill disease": "bgd",
               "Bacterial red disease": "brd", "EUS": "eus"}
SPECIES_KEYS = {"Catla": "catla", "Rohu": "rohu", "Mrigal": "mrigal"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
SEQ = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # one-hue blue ramp, zero = paper
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIG_H = 1.89                                         # Figure 3 panel height in inches (4.8 cm), drawn at print size


def pct(x):  return None if x is None else f"{100.0 * x:.1f}"
def auc(x):  return None if x is None else f"{x:.3f}"
def ms(x):   return None if x is None else f"{x:.1f}"
def pts(x, signed=False): return (f"{x:+.1f}" if signed else f"{x:.1f}").replace("-", "\u2212")  # real minus sign
def rng_(lo, hi): return f"{pts(lo)} to {pts(hi)}"


# ---------------- fish-level bootstrap -------------------------------------------------------
# The split unit is the fish (MatsyaDx specimen, or a near-duplicate group), so confidence intervals
# resample FISH, not images: with ~13 health test fish an image-level bootstrap would be far too narrow.
# Fish are resampled within strata (health label, or species for species-only images) so every class
# stays present, and the same resamples are used for every b, which makes the b=1 vs b=4 comparison paired.

def _wacc(y, p, w):
    return float((w * (y == p)).sum() / w.sum())


def _wf1(y, p, w, n):
    """macro-F1 with sample weights; with w = 1 it equals sklearn's f1_score(average='macro')."""
    hit = y == p
    tp = np.bincount(y[hit], weights=w[hit], minlength=n)
    t, q = np.bincount(y, weights=w, minlength=n), np.bincount(p, weights=w, minlength=n)
    used = (t > 0) | (q > 0)
    return float((2 * tp[used] / (t[used] + q[used])).mean())


def pool_cv(cv_dir, healthy, bs=(1, 2, 4), tags=None):
    """Out-of-fold predictions: every image is predicted once, by the fold model that did not train on it.
    tags maps b to a run-name suffix, e.g. {4: "up"} pools runs/cv/b4up_f* for b = 4."""
    out, folds = {}, None
    for b in bs:
        tag = (tags or {}).get(b, "")
        fs = sorted(glob.glob(os.path.join(cv_dir, f"b{b}{tag}_f*", "test_scores.npz")))
        if not fs:
            continue
        ks = sorted(os.path.basename(os.path.dirname(f)).split("_f")[1] for f in fs)
        folds = ks if folds is None else folds
        if ks != folds:                                   # pool only when every b has the same folds
            print(f"cv: b={b} has folds {ks}, others {folds}; skipped")
            continue
        d = [np.load(f) for f in fs]
        idx = np.concatenate([x["idx"] for x in d]); o = np.argsort(idx)
        cat = lambda k: np.concatenate([x[k] for x in d])[o]
        out[b] = (idx[o], cat("ps").argmax(1), cat("ph").argmax(1), cat("s_md"), cat("s_msp"))
    return out, (len(folds) if folds else 0)


def fish_bootstrap(cache, runs_dir, bs=(1, 2, 4), B=2000, seed=0, scores=None):
    import csv
    from sklearn.metrics import roc_auc_score
    rows = list(csv.DictReader(open(os.path.join(cache, "manifest.csv"), encoding="utf-8")))
    vocab = json.load(open(os.path.join(cache, "vocab.json"), encoding="utf-8"))
    healthy = vocab["health"].index("Healthy")
    sp = np.array([int(r["species"]) for r in rows]); he = np.array([int(r["health"]) for r in rows])
    grp = np.array([r["group"] for r in rows]); src = np.array([r["source"] for r in rows])
    P, idx = {}, None
    for b in bs:
        if scores is not None:                            # pooled cross-validation predictions
            if b not in scores:
                continue
            ix, ps_, ph_, smd_, smsp_ = scores[b]
        else:
            f = os.path.join(runs_dir, f"b{b}", "test_scores.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            ix, ps_, ph_, smd_ = d["idx"], d["ps"].argmax(1), d["ph"].argmax(1), d["s_md"]
            smsp_ = d["s_msp"] if "s_msp" in d else 1.0 - d["ph"][:, healthy]
        if idx is None:
            idx = ix
        assert np.array_equal(idx, ix), "runs were evaluated on different test sets"
        P[b] = (ps_, ph_, smd_, smsp_)
    if not P:
        return None
    ys, yh, g = sp[idx], he[idx], grp[idx]
    ms_, mh = ys >= 0, yh >= 0
    # same-image check: the farm photographs carry BOTH labels, so species and health can be compared on
    # identical images (accuracy, because only a few species occur in them and macro-F1 would be skewed)
    mb = ms_ & mh & (src[idx] == "matsya")
    ugrp, gi = np.unique(g, return_inverse=True)
    first = np.zeros(len(ugrp), int); first[gi[::-1]] = np.arange(len(gi))[::-1]
    stratum = np.where(yh[first] >= 0, yh[first], 100 + ys[first])
    strata = [np.nonzero(stratum == s)[0] for s in np.unique(stratum)]
    nS, nH = len(vocab["species"]), len(vocab["health"])

    def metrics(w):
        out = {}
        for b, (ps, ph, smd, smsp) in P.items():
            out[f"sp_acc_b{b}"] = _wacc(ys[ms_], ps[ms_], w[ms_])
            out[f"sp_f1_b{b}"] = _wf1(ys[ms_], ps[ms_], w[ms_], nS)
            out[f"he_acc_b{b}"] = _wacc(yh[mh], ph[mh], w[mh])
            out[f"he_f1_b{b}"] = _wf1(yh[mh], ph[mh], w[mh], nH)
            dis = (yh[mh] != healthy).astype(int)
            ok = 0 < dis.sum() < len(dis)
            out[f"auc_md_b{b}"] = float(roc_auc_score(dis, smd[mh], sample_weight=w[mh])) if ok else np.nan
            out[f"auc_msp_b{b}"] = float(roc_auc_score(dis, smsp[mh], sample_weight=w[mh])) if ok else np.nan
            pd_ = ph[mh] != healthy                       # binary sick/healthy from the health head
            out[f"sens_b{b}"] = float((w[mh] * (pd_ & (dis == 1))).sum() / max((w[mh] * (dis == 1)).sum(), 1e-12))
            out[f"spec_b{b}"] = float((w[mh] * (~pd_ & (dis == 0))).sum() / max((w[mh] * (dis == 0)).sum(), 1e-12))
            md_ = mh & (yh != healthy)                    # disease TYPE among truly diseased fish
            out[f"dtype_acc_b{b}"] = _wacc(yh[md_], ph[md_], w[md_]) if md_.any() else np.nan
            if mb.any():
                out[f"si_sp_acc_b{b}"] = _wacc(ys[mb], ps[mb], w[mb])
                out[f"si_he_acc_b{b}"] = _wacc(yh[mb], ph[mb], w[mb])
        if 1 in P and 4 in P:
            out["dsp_f1"] = out["sp_f1_b4"] - out["sp_f1_b1"]      # change from b=1 to b=4 (negative = loss)
            out["dhe_f1"] = out["he_f1_b4"] - out["he_f1_b1"]
            out["ddrop"] = out["dhe_f1"] - out["dsp_f1"]           # negative = health loses more than species
            if mb.any():                                           # same images, accuracy
                out["si_dsp"] = out["si_sp_acc_b4"] - out["si_sp_acc_b1"]
                out["si_dhe"] = out["si_he_acc_b4"] - out["si_he_acc_b1"]
                out["si_ddrop"] = out["si_dhe"] - out["si_dsp"]
        return out

    point = metrics(np.ones(len(idx)))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(B):
        cnt = np.zeros(len(ugrp))
        for s in strata:
            np.add.at(cnt, rng.choice(s, size=len(s), replace=True), 1)
        for k, v in metrics(cnt[gi]).items():
            draws[k].append(v)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in draws.items()}
    n_fish = len(set(g[src[idx] == "matsya"]))
    n_healthy_fish = len(set(g[(src[idx] == "matsya") & (yh == healthy)]))
    return {"point": point, "ci": ci, "B": B, "n_test_fish": n_fish, "n_healthy_fish": n_healthy_fish,
            "n_strata": len(strata), "n_same_image": int(mb.sum())}


def write_bootstrap(R, BS, pre=""):
    """Point values and 95% intervals as paper macros; pre = "" (main split) or "cv_" (cross-validation)."""
    for k, (lo, hi) in BS["ci"].items():
        v = BS["point"][k]
        if k.startswith(("sp_", "he_", "si_sp_", "si_he_", "sens_", "spec_", "dtype_")):
            R[pre + k], R[pre + k + "_ci"] = pct(v), f"{100 * lo:.1f}\u2013{100 * hi:.1f}"
        elif k.startswith("auc_"):
            R[pre + k], R[pre + k + "_ci"] = auc(v), f"{lo:.3f}\u2013{hi:.3f}"
        else:                                   # dsp_f1, dhe_f1, ddrop, si_d*: changes in points
            R[pre + k], R[pre + k + "_ci"] = pts(100 * v, signed=True), rng_(100 * lo, 100 * hi)
            R[pre + k + "_abs"] = pts(abs(100 * v))          # size of the change, for prose
    R[pre + "n_same_image"] = f"{BS['n_same_image']:,}"
    P = BS["point"]                              # relative losses b=1 -> b=4, in % of the b=1 macro-F1
    if all(k in P for k in ("dsp_f1", "dhe_f1", "sp_f1_b1", "he_f1_b1")):
        R[pre + "rel_sp"] = f"{100 * abs(P['dsp_f1']) / P['sp_f1_b1']:.0f}"
        R[pre + "rel_he"] = f"{100 * abs(P['dhe_f1']) / P['he_f1_b1']:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cost", default="model_cost.json", help="from model_cost.py")
    ap.add_argument("--paper", default="results")
    a = ap.parse_args()
    R = {}

    # ---- dataset -----------------------------------------------------------------
    sp = os.path.join(a.cache, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp, encoding="utf-8"))
        R["n_img"], R["n_groups"] = f"{s['n_images']:,}", f"{s['n_groups']:,}"
        for src, d in s["by_source_split"].items():
            for k in ("train", "val", "test"):
                R[f"{src}_{k}"] = f"{d.get(k, 0):,}"
            R[f"{src}_all"] = f"{sum(d.values()):,}"
        for k in ("train", "val", "test"):
            R[f"n_{k}"] = f"{sum(d.get(k, 0) for d in s['by_source_split'].values()):,}"
        for name, d in s["species"].items():
            R[f"cnt_sp_{name.lower().replace(' ', '').replace('-', '')}"] = str(sum(d.values()))
        if "detector_found_rate" in s:
            R["det_found"] = pct(s["detector_found_rate"])
    man = os.path.join(a.cache, "manifest.csv")
    if os.path.exists(man):
        import csv
        R["n_test_fish"] = str(len({r["group"] for r in csv.DictReader(open(man, encoding="utf-8"))
                                    if r["split"] == "test" and r["source"] == "matsya"}))

    # ---- runs --------------------------------------------------------------------
    runs = {}
    for f in glob.glob(os.path.join(a.runs, "*", "metrics.json")):
        m = json.load(open(f, encoding="utf-8"))
        runs[os.path.basename(os.path.dirname(f))] = m
    for b in (1, 2, 4):
        m = runs.get(f"b{b}")
        if not m:
            continue
        t = f"b{b}"
        R[f"sp_acc_{t}"], R[f"sp_f1_{t}"] = pct(m["species"]["acc"]), pct(m["species"]["macro_f1"])
        R[f"sp_dis_{t}"] = pct(m["species"].get("acc_on_diseased_fish"))
        R[f"he_acc_{t}"], R[f"he_f1_{t}"] = pct(m["health"]["acc"]), pct(m["health"]["macro_f1"])
        R[f"he_sens_{t}"], R[f"he_spec_{t}"] = pct(m["health"]["sensitivity"]), pct(m["health"]["specificity"])
        sv = m["abnormality"]["seen_vs_healthy"]
        R[f"auc_md_{t}"], R[f"auc_msp_{t}"] = auc(sv["auroc_md"]), auc(sv["auroc_msp"])
        R[f"fa_{t}"] = pct(m["abnormality"]["false_alarm_on_healthy_test"])
        R[f"fam_{t}"] = pct(m["abnormality"].get("false_alarm_msp_rule"))
        tfl = os.path.join(a.runs, t, "tflite.json")
        if os.path.exists(tfl):
            q = json.load(open(tfl, encoding="utf-8"))
            R[f"size_int8_{t}"] = f"{q['int8']['size_mb']:.2f}"
            R[f"spq_{t}"], R[f"heq_{t}"] = pct(q["int8"]["species_acc"]), pct(q["int8"]["health_acc"])
        if b == 2:
            for name, key in HEALTH_KEYS.items():
                pc = m["health"]["per_class"].get(name)
                if pc:
                    R[f"hp_{key}"], R[f"hr_{key}"], R[f"hf_{key}"] = pct(pc["precision"]), pct(pc["recall"]), pct(pc["f1"])
            for name, key in SPECIES_KEYS.items():
                pc = m["species"]["per_class"].get(name)
                if pc:
                    R[f"spf_{key}"] = pct(pc["f1"])
    lodo_md, lodo_msp, lodo_flag, lodo_flagm, lodo_fa, lodo_fam = [], [], [], [], [], []
    for name, key in HEALTH_KEYS.items():
        m = runs.get(f"b2_lodo_{name.replace(' ', '_')}")
        if not m or "unseen_vs_healthy" not in m["abnormality"]:
            continue
        u = m["abnormality"]["unseen_vs_healthy"]
        R[f"lodo_md_{key}"], R[f"lodo_msp_{key}"] = auc(u["auroc_md"]), auc(u["auroc_msp"])
        R[f"lodo_fpr_{key}"], R[f"lodo_fprm_{key}"] = pct(u["fpr95_md"]), pct(u["fpr95_msp"])
        R[f"lodo_nfish_{key}"] = str(u.get("n_fish", ""))
        R[f"lodo_flagm_{key}"] = pct(m["abnormality"].get("unseen_flag_rate_msp_rule"))
        if m["abnormality"].get("unseen_flag_rate_msp_rule") is not None:
            lodo_flagm.append(m["abnormality"]["unseen_flag_rate_msp_rule"])
        R[f"lodo_flag_{key}"] = pct(m["abnormality"]["unseen_flag_rate_rule"])
        R[f"lodo_head_{key}"] = pct(m["abnormality"]["unseen_flag_rate_head_only"])
        lodo_md.append(u["auroc_md"]); lodo_msp.append(u["auroc_msp"])
        lodo_flag.append(m["abnormality"]["unseen_flag_rate_rule"])
        # false alarms of the SAME held-out model on healthy test fish, so each flag rate has its own cost
        fa, fam = m["abnormality"].get("false_alarm_on_healthy_test"), m["abnormality"].get("false_alarm_msp_rule")
        if fa is not None:
            R[f"lodo_fa_{key}"] = pct(fa); lodo_fa.append(fa)
        if fam is not None:
            R[f"lodo_fam_{key}"] = pct(fam); lodo_fam.append(fam)
    if lodo_md:
        R["lodo_md_mean"], R["lodo_msp_mean"] = auc(np.mean(lodo_md)), auc(np.mean(lodo_msp))
        R["lodo_flag_mean"] = pct(np.mean(lodo_flag))
        if len(lodo_flagm) == len(lodo_flag):
            R["lodo_flagm_mean"] = pct(np.mean(lodo_flagm))
        if len(lodo_fa) == len(lodo_flag):
            R["lodo_fa_mean"] = pct(np.mean(lodo_fa))
        if len(lodo_fam) == len(lodo_flag):
            R["lodo_fam_mean"] = pct(np.mean(lodo_fam))

    # ---- fish-level bootstrap CIs (paired across b) ---------------------------------------
    BS = None
    try:
        BS = fish_bootstrap(a.cache, a.runs)
    except Exception as e:  # missing cache / runs -> numbers stay as placeholders
        print("bootstrap skipped:", e)
    if BS:
        write_bootstrap(R, BS)
        for b in (1, 2, 4):                         # the bootstrap's own point estimate must match evaluate.py
            m = runs.get(f"b{b}")
            if m and f"sp_f1_b{b}" in BS["point"]:
                for k, ref in ((f"sp_f1_b{b}", m["species"]["macro_f1"]), (f"he_f1_b{b}", m["health"]["macro_f1"])):
                    if abs(BS["point"][k] - ref) > 1e-6:
                        print(f"WARNING: {k} bootstrap point {BS['point'][k]:.4f} != metrics.json {ref:.4f}")
        json.dump(BS, open(os.path.join(a.runs, "bootstrap.json"), "w", encoding="utf-8"), indent=2)
        print(f"bootstrap: {BS['B']} fish-level resamples, {BS['n_strata']} strata -> {a.runs}/bootstrap.json")

    # ---- cross-validation and controls (run_all.py MODE = "extra" / "upcv") -----------------------------------------
    cv_dir = os.path.join(a.runs, "cv")
    if os.path.isdir(cv_dir):
        vocab = json.load(open(os.path.join(a.cache, "vocab.json"), encoding="utf-8"))
        pooled, K = pool_cv(cv_dir, vocab["health"].index("Healthy"))
        if pooled:
            CV = fish_bootstrap(a.cache, None, scores=pooled)
            write_bootstrap(R, CV, "cv_")
            R["cv_k"], R["cv_n_fish"], R["cv_n_healthy_fish"] = str(K), str(CV["n_test_fish"]), str(CV["n_healthy_fish"])
            json.dump(CV, open(os.path.join(a.runs, "bootstrap_cv.json"), "w", encoding="utf-8"), indent=2)
            print(f"cross-validation: {K} folds pooled, {CV['n_test_fish']} farm fish ({CV['n_healthy_fish']} healthy)")
        pooled_up, K_up = pool_cv(cv_dir, vocab["health"].index("Healthy"), bs=(1, 4), tags={4: "up"})
        if 1 in pooled_up and 4 in pooled_up:         # MODE = "upcv": enlarged-input control, cross-validated
            CVU = fish_bootstrap(a.cache, None, scores=pooled_up)
            write_bootstrap(R, CVU, "cvup_")
            for key, name in (("sp_f1_b4", "sp"), ("he_f1_b4", "he")):   # what enlarging adds, cross-validated
                g = 100 * (CVU["point"][key] - CV["point"][key])
                R[f"cvup_gain_{name}_b4"], R[f"cvup_gain_{name}_b4_abs"] = pts(g, signed=True), pts(abs(g))
            json.dump(CVU, open(os.path.join(a.runs, "bootstrap_cvup.json"), "w", encoding="utf-8"), indent=2)
            print(f"enlarged control, cross-validated: {K_up} folds pooled")
    for b in (2, 4):                                  # enlarged-input control on the main split
        m, m0 = runs.get(f"b{b}_up"), runs.get(f"b{b}")
        if not m:
            continue
        for k, v in (("sp_acc", m["species"]["acc"]), ("sp_f1", m["species"]["macro_f1"]),
                     ("he_acc", m["health"]["acc"]), ("he_f1", m["health"]["macro_f1"])):
            R[f"up_{k}_b{b}"] = pct(v)
        R[f"up_auc_msp_b{b}"] = auc(m["abnormality"]["seen_vs_healthy"]["auroc_msp"])
        if m0:
            R[f"up_gain_sp_b{b}"] = pts(100 * (m["species"]["macro_f1"] - m0["species"]["macro_f1"]), signed=True)
            R[f"up_gain_he_b{b}"] = pts(100 * (m["health"]["macro_f1"] - m0["health"]["macro_f1"]), signed=True)
            R[f"up_gain_sp_b{b}_abs"] = pts(abs(100 * (m["species"]["macro_f1"] - m0["species"]["macro_f1"])))
            R[f"up_gain_he_b{b}_abs"] = pts(abs(100 * (m["health"]["macro_f1"] - m0["health"]["macro_f1"])))
    # bootstrap of the enlarged control: b=1 against the enlarged b=2 / b=4 models, same test fish
    up_scores = {}
    for b, name in ((1, "b1"), (2, "b2_up"), (4, "b4_up")):
        f = os.path.join(a.runs, name, "test_scores.npz")
        if os.path.exists(f):
            d = np.load(f)
            up_scores[b] = (d["idx"], d["ps"].argmax(1), d["ph"].argmax(1), d["s_md"], d["s_msp"])
    if 1 in up_scores and 4 in up_scores:
        UP = fish_bootstrap(a.cache, a.runs, scores=up_scores)
        if UP:
            write_bootstrap(R, UP, "upbs_")
    extra = {}
    for name, key in HEALTH_KEYS.items():             # stronger abnormality scores on the held-out diseases
        m = runs.get(f"b2_lodo_{name.replace(' ', '_')}")
        u = m["abnormality"].get("unseen_vs_healthy", {}) if m else {}
        for sc in ("energy", "maxsm", "knn", "pcamd"):
            if u.get(f"auroc_{sc}") is not None:
                R[f"lodo_{sc}_{key}"] = auc(u[f"auroc_{sc}"])
                R[f"lodo_fpr{sc}_{key}"] = pct(u.get(f"fpr95_{sc}"))
                extra.setdefault(sc, []).append(u[f"auroc_{sc}"])
    for sc, v in extra.items():
        if len(v) == 3:
            R[f"lodo_{sc}_mean"] = auc(np.mean(v))

    # ---- software model cost (model_cost.py) -----------------------------------------
    if os.path.exists(a.cost):
        c = json.load(open(a.cost, encoding="utf-8"))
        R["params_m"] = f"{c['b2']['params'] / 1e6:.2f}"
        for b in (1, 2, 4):
            if f"b{b}" in c:
                R[f"mflops_b{b}"] = f"{c[f'b{b}']['flops'] / 1e6:.1f}"

    R = {k: v for k, v in R.items() if v is not None}
    os.makedirs(os.path.join(a.paper, "figures"), exist_ok=True)
    with open(os.path.join(a.paper, "results_macros.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by make_paper_numbers.py -- do not edit by hand\n")
        for k in sorted(R):
            f.write(f"\\resdef{{{k}}}{{{R[k]}}}\n")
    with open(os.path.join(a.paper, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Results\n\n")
        for k in sorted(R):
            f.write(f"- `{k}` = {R[k]}\n")
    print(f"wrote {len(R)} values to {a.paper}/results_macros.tex")

    # ---- figures --------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    bs = [b for b in (1, 2, 4) if f"b{b}" in runs]
    if bs:
        series = [("Species F1", [runs[f"b{b}"]["species"]["macro_f1"] * 100 for b in bs], "o"),
                  ("Health F1", [runs[f"b{b}"]["health"]["macro_f1"] * 100 for b in bs], "s"),
                  ("Sick-vs-healthy AUROC", [(runs[f"b{b}"]["abnormality"]["seen_vs_healthy"]["auroc_msp"] or 0) * 100 for b in bs], "^")]
        fig, ax = plt.subplots(figsize=(3.05, FIG_H), dpi=300)
        xs = [100.0 / b**2 for b in bs]
        ci_keys = ["sp_f1", "he_f1", "auc_msp"]
        for (lab, ys, mk), c, ck, dx in zip(series, COLORS, ci_keys, (0.93, 1.0, 1.075)):
            xo = [x * dx for x in xs]               # small horizontal stagger so the error bars do not overlap
            ax.plot(xo, ys, color=c, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.8, label=lab, zorder=3)
            if BS:                                  # 95% fish-level bootstrap interval
                lo = [100 * BS["ci"].get(f"{ck}_b{b}", (np.nan, np.nan))[0] for b in bs]
                hi = [100 * BS["ci"].get(f"{ck}_b{b}", (np.nan, np.nan))[1] for b in bs]
                err = np.clip([np.subtract(ys, lo), np.subtract(hi, ys)], 0, None)
                ax.errorbar(xo, ys, yerr=err, fmt="none", ecolor=c, elinewidth=0.9, capsize=2.2, zorder=2)
        # end-of-line value labels, nudged apart so they never overlap
        ends = sorted(((ys[-1], i) for i, (_, ys, _) in enumerate(series)))
        lo, hi = min(min(s[1]) for s in series), max(max(s[1]) for s in series)
        y0, y1 = ax.get_ylim()
        gap = 0.075 * (y1 - y0)                     # one label height in axis units
        placed = []
        for y, i in ends:
            yy = y if not placed or y - placed[-1] >= gap else placed[-1] + gap
            placed.append(yy)
            ax.annotate(f"{y:.1f}", (xs[-1], y), xytext=(xs[-1] / 1.25, yy), textcoords="data",
                        ha="left", va="center", fontsize=7, color=INK)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
        ax.set_xlim(xs[0] * 1.6, xs[-1] / 2.2)
        ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}%\n(b={b})" for x, b in zip(xs, bs)])
        ax.set_xlabel("Measurements kept per frame"); ax.set_ylabel("Score (%)")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                  handlelength=1.6, columnspacing=0.9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.paper, "figures", f"fig_compression.{ext}"), bbox_inches="tight")
        plt.close(fig)

    # health confusion matrix at b = 2: counts in the cells, colour = share of the true class
    m2 = runs.get("b2")
    if m2 and m2["health"].get("confusion"):
        from matplotlib.colors import LinearSegmentedColormap
        C = np.array(m2["health"]["confusion"], float)
        share = C / np.maximum(C.sum(1, keepdims=True), 1)
        lab = ["Healthy", "BGD", "BRD", "EUS"]
        fig, ax = plt.subplots(figsize=(2.1, FIG_H), dpi=300)
        ax.imshow(share, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
        for i in range(len(lab)):
            for j in range(len(lab)):
                ax.text(j, i, f"{int(C[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] >= 0.5 else INK)
        ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
        ax.set_xticks(np.arange(-0.5, len(lab)), minor=True); ax.set_yticks(np.arange(-0.5, len(lab)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="both", length=0)
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        fig.tight_layout(pad=0.2)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.paper, "figures", f"fig_confusion.{ext}"), bbox_inches="tight")
        plt.close(fig)

    # one test fish at b = 1, 2, 4 (prefer a diseased MatsyaDx fish)
    try:
        import csv
        from dbbd import dbbd_fast
        imgs = np.load(os.path.join(a.cache, "images_224.npy"), mmap_mode="r")
        rows = list(csv.DictReader(open(os.path.join(a.cache, "manifest.csv"), encoding="utf-8")))
        vocab = json.load(open(os.path.join(a.cache, "vocab.json"), encoding="utf-8"))
        eus = vocab["health"].index("EUS")
        pick = next((int(r["idx"]) for r in rows if r["split"] == "test" and int(r["health"]) == eus),
                    int(rows[0]["idx"]))
        x = np.asarray(imgs[pick])
        from PIL import Image
        for b in (1, 2, 4):                          # bare panels for Figure 1, enlarged without smoothing
            Image.fromarray(dbbd_fast(x, b).astype(np.uint8)).resize((448, 448), Image.NEAREST).save(
                os.path.join(a.paper, "figures", f"fish_b{b}.png"))
        fig, axs = plt.subplots(1, 3, figsize=(4.6, 1.9), dpi=300)
        for ax, b in zip(axs, (1, 2, 4)):
            ax.imshow(dbbd_fast(x, b), interpolation="nearest"); ax.set_axis_off()
            ax.set_title(f"b={b}: {224 // b}x{224 // b} ({100 / b**2:g}%)", fontsize=7, color=INK)
        fig.tight_layout(pad=0.4)
        fig.savefig(os.path.join(a.paper, "figures", "fig_dbbd_levels.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # cache missing -> keep the placeholder
        print("fig_dbbd_levels skipped:", e)


if __name__ == "__main__":
    main()
