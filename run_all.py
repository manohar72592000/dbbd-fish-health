"""
Run all experiments from Spyder or any Python, on Windows, Linux or macOS.

    1. Set the paths below.
    2. MODE = "check" first  -> 3-5 min test on fake data; proves the environment works.
    3. MODE = "full"         -> all experiments (about 2-4 h on a desktop CPU).
    4. Press F5 (Run file).

The full run can be stopped at any time and restarted: finished steps are skipped.
Each step runs as a separate Python process so memory is released between runs; its output is shown
live in the console and also saved to runs/run_log.txt.

CPU is used on purpose. TensorFlow has no GPU support on native Windows after version 2.10, and its
official builds do not include kernels for RTX 50-series (Blackwell) cards. MobileNetV2 is small enough
that the CPU finishes in a few hours.
"""
MODE = "check"                                   # "check", "full", "reeval" (re-score only), "extra" (cross-validation,
                                                 # input-size control, extra scores) or "upcv" (input-size control in CV)
BDFF = r"path/to/BD-Freshwater-Fish"             # folder that contains the 12 species folders
MATSYA = r"path/to/MatsyaDx-BD"                  # folder that contains the MatsyaDx-BD images
DET = r""                                        # optional: your DBBD-YOLO best.pt (needs ultralytics)

# expected image counts (from the dataset pages) -- used to catch a wrong folder before training
EXPECT = {"bdff": 4389, "matsya": 2137,
          "health": {"Healthy": 877, "Bacterial red disease": 653, "EUS": 332, "Bacterial gill disease": 275}}

import json, os, shutil, subprocess, sys, tempfile, time, zipfile

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                 # pasted into a console instead of run as a file
    HERE = os.getcwd()
RUNS, CACHE, PAPER = os.path.join(HERE, "runs"), os.path.join(HERE, "data_cache"), os.path.join(HERE, "results")
LOG = None


def env():
    e = dict(os.environ)
    e.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2",
             CUDA_VISIBLE_DEVICES="-1")       # CPU only (see note at the top)
    return e


def step(title, args, done=None):
    """Run `python <args>` in the kit folder unless `done` already exists."""
    if done and os.path.exists(done):
        print(f"[skip] {title} (already done)")
        return
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", *args], cwd=HERE, env=env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        if "WARNING: All log messages before absl" in line or "oneDNN custom operations" in line:
            continue
        print(line, end="", flush=True)
        if LOG:
            LOG.write(line); LOG.flush()
    if p.wait() != 0:
        raise RuntimeError(f"step failed: {title}. Scroll up (or open {LOG.name if LOG else 'the log'}) for the error.")
    msg = f"--- {title}: done in {(time.time() - t0) / 60:.1f} min"
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n"); LOG.flush()


def check_data():
    """Stop before training if the datasets were not found where expected."""
    s = json.load(open(os.path.join(CACHE, "summary.json"), encoding="utf-8"))
    src = {k: sum(v.values()) for k, v in s["by_source_split"].items()}
    hea = {k: sum(v.values()) for k, v in s.get("health", {}).items()}
    print(f"\nImages found: BD-Freshwater-Fish {src.get('bdff', 0)} (expected {EXPECT['bdff']}), "
          f"MatsyaDx-BD {src.get('matsya', 0)} (expected {EXPECT['matsya']})")
    print("Health classes:", ", ".join(f"{k} {hea.get(k, 0)} (exp. {v})" for k, v in EXPECT["health"].items()))
    print(f"Fish groups: {s['n_groups']}, near-duplicate merges: {s.get('near_duplicate_merges')}")
    import csv
    rows = list(csv.DictReader(open(os.path.join(CACHE, "manifest.csv"), encoding="utf-8")))
    n_fish = len({r["group"] for r in rows if r["source"] == "matsya"})
    print(f"MatsyaDx fish (specimen groups): {n_fish} (expected about 86)")
    if n_fish > 150:
        raise RuntimeError("MatsyaDx images are not grouped by fish. This looks like version 1 (class folders only). "
                           "Download version 2 (specimen folders, doi 10.17632/sxkynv9t7n.2), delete data_cache, run again.")
    bad = [k for k in ("bdff", "matsya") if src.get(k, 0) < 0.9 * EXPECT[k]]
    bad += [k for k, v in EXPECT["health"].items() if hea.get(k, 0) < 0.9 * v]
    if bad:
        raise RuntimeError(f"Too few images for: {bad}. Check BDFF / MATSYA paths, delete the data_cache "
                           f"folder, and run again. Labels that could not be read are listed as [skip] above.")


def full():
    global LOG
    os.makedirs(RUNS, exist_ok=True)
    LOG = open(os.path.join(RUNS, "run_log.txt"), "a", encoding="utf-8")
    LOG.write(f"\n##### run started {time.ctime()} #####\n")
    t0 = time.time()
    for p in (BDFF, MATSYA):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Folder not found: {p}  (edit the paths at the top of run_all.py)")

    step("0. model size and MFLOPs", ["model_cost.py", "--out", "model_cost.json"], "model_cost.json")
    prep = ["prepare_data.py", "--bdff_dir", BDFF, "--matsya_dir", MATSYA, "--out", CACHE]
    if DET:
        prep += ["--detector", DET]
    step("1. prepare data (resize, near-duplicate grouping, fish-level split)", prep,
         os.path.join(CACHE, "manifest.csv"))
    check_data()

    for b in (2, 1, 4):                       # deployed setting first, then the two ends
        out = os.path.join(RUNS, f"b{b}")
        step(f"2. train b={b}", ["train.py", "--cache", CACHE, "--b", str(b), "--out", out],
             os.path.join(out, "best.keras"))
        step(f"2. evaluate b={b}", ["evaluate.py", "--cache", CACHE, "--run", out], os.path.join(out, "metrics.json"))
        step(f"2. int8 TFLite b={b}", ["export_tflite.py", "--cache", CACHE, "--run", out],
             os.path.join(out, "tflite.json"))

    for d in ("EUS", "Bacterial red disease", "Bacterial gill disease"):
        out = os.path.join(RUNS, "b2_lodo_" + d.replace(" ", "_"))
        step(f"3. leave out {d}: train", ["train.py", "--cache", CACHE, "--b", "2", "--holdout", d, "--out", out],
             os.path.join(out, "best.keras"))
        step(f"3. leave out {d}: evaluate", ["evaluate.py", "--cache", CACHE, "--run", out],
             os.path.join(out, "metrics.json"))

    step("4. model cost with the real class counts", ["model_cost.py", "--cache", CACHE, "--out", "model_cost.json"])
    step("5. tables, intervals and figures",
         ["make_paper_numbers.py", "--cache", CACHE, "--runs", RUNS, "--cost", "model_cost.json", "--paper", PAPER])

    bundle()
    print(f"\nALL DONE in {(time.time() - t0) / 3600:.1f} h.")


def bundle():
    """Small zip of metrics, scores and generated tables (no images, no models)."""
    zp = os.path.join(HERE, "results_bundle.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(RUNS):
            for f in files:
                if f.endswith((".json", ".csv", ".txt", ".npz")):
                    z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), HERE))
        for f in ("manifest.csv", "vocab.json", "summary.json", "folds.json"):
            if os.path.exists(os.path.join(CACHE, f)):
                z.write(os.path.join(CACHE, f), os.path.join("data_cache", f))
        z.write(os.path.join(HERE, "model_cost.json"), "model_cost.json")
        for f in ("results_macros.tex", "results_summary.md", "figures/fig_compression.pdf",
                  "figures/fig_compression.png", "figures/fig_confusion.pdf", "figures/fig_confusion.png",
                  "figures/fish_b1.png", "figures/fish_b2.png", "figures/fish_b4.png",
                  "figures/fig_dbbd_levels.png"):
            fp = os.path.join(PAPER, f)
            if os.path.exists(fp):
                z.write(fp, os.path.join("results", f))
    print(f"Results bundled in {zp}")


def extra():
    """Robustness checks (about 2-2.5 h on a desktop CPU; restartable like "full"):
       A. 5-fold cross-validation by fish at b = 1, 2, 4, so every fish, including all 10 healthy ones,
          is tested once by a model that never saw it;
       B. control: the b = 2 and b = 4 DBBD images enlarged back to 224 x 224 (no new information), to
          separate lost information from the smaller network input;
       C. re-scoring the leave-one-disease-out models with stronger abnormality scores."""
    global LOG
    os.makedirs(RUNS, exist_ok=True)
    LOG = open(os.path.join(RUNS, "run_log.txt"), "a", encoding="utf-8")
    LOG.write(f"\n##### extra run started {time.ctime()} #####\n")
    t0 = time.time()
    if not os.path.exists(os.path.join(CACHE, "manifest.csv")):
        raise FileNotFoundError(f"No data cache in {CACHE}. Run MODE = \"full\" first.")
    folds = os.path.join(CACHE, "folds.json")
    step("A0. fish-level folds", ["make_folds.py", "--cache", CACHE, "--k", "5"], folds)
    for b in (2, 4, 1):
        for k in range(5):
            out = os.path.join(RUNS, "cv", f"b{b}_f{k}")
            step(f"A. fold {k}, b={b}: train", ["train.py", "--cache", CACHE, "--b", str(b), "--folds", folds,
                                                "--fold", str(k), "--seed", str(k), "--out", out],
                 os.path.join(out, "best.keras"))
            step(f"A. fold {k}, b={b}: evaluate", ["evaluate.py", "--cache", CACHE, "--run", out],
                 os.path.join(out, "metrics.json"))
    for b in (2, 4):
        out = os.path.join(RUNS, f"b{b}_up")
        step(f"B. enlarged control b={b}: train", ["train.py", "--cache", CACHE, "--b", str(b), "--upsample", "--out", out],
             os.path.join(out, "best.keras"))
        step(f"B. enlarged control b={b}: evaluate", ["evaluate.py", "--cache", CACHE, "--run", out],
             os.path.join(out, "metrics.json"))
    for d in sorted(x for x in os.listdir(RUNS) if x.startswith("b2_lodo_")):
        step(f"C. re-score {d}", ["evaluate.py", "--cache", CACHE, "--run", os.path.join(RUNS, d)])
    step("tables, intervals and figures",
         ["make_paper_numbers.py", "--cache", CACHE, "--runs", RUNS, "--cost", "model_cost.json", "--paper", PAPER])
    bundle()
    print(f"\nEXTRA DONE in {(time.time() - t0) / 3600:.1f} h.")


def upcv():
    """Input-size control in cross-validation (about 1 h on a desktop CPU; restartable): b = 4 enlarged
       in the same 5 folds as the cross-validation, so it is compared with the b = 1 fold models on all 86 fish."""
    global LOG
    LOG = open(os.path.join(RUNS, "run_log.txt"), "a", encoding="utf-8")
    LOG.write(f"\n##### upcv run started {time.ctime()} #####\n")
    t0 = time.time()
    folds = os.path.join(CACHE, "folds.json")
    if not os.path.exists(folds) or not os.path.isdir(os.path.join(RUNS, "cv")):
        raise FileNotFoundError("No cross-validation runs found. Run MODE = \"extra\" first.")
    for k in range(5):
        out = os.path.join(RUNS, "cv", f"b4up_f{k}")
        step(f"D. fold {k}, b=4 enlarged: train", ["train.py", "--cache", CACHE, "--b", "4", "--upsample",
                                                    "--folds", folds, "--fold", str(k), "--seed", str(k), "--out", out],
             os.path.join(out, "best.keras"))
        step(f"D. fold {k}, b=4 enlarged: evaluate", ["evaluate.py", "--cache", CACHE, "--run", out],
             os.path.join(out, "metrics.json"))
    step("tables, intervals and figures",
         ["make_paper_numbers.py", "--cache", CACHE, "--runs", RUNS, "--cost", "model_cost.json", "--paper", PAPER])
    bundle()
    print(f"\nUPCV DONE in {(time.time() - t0) / 3600:.1f} h.")


def reeval():
    """Re-run only the evaluation of the already-trained models (a few minutes, no training)."""
    global LOG
    LOG = open(os.path.join(RUNS, "run_log.txt"), "a", encoding="utf-8")
    runs = sorted(d for d in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, d, "best.keras")))
    if not runs:
        raise FileNotFoundError(f"No trained models in {RUNS}. Run MODE = \"full\" first.")
    for d in runs:
        step(f"re-evaluate {d}", ["evaluate.py", "--cache", CACHE, "--run", os.path.join(RUNS, d)])
    step("tables, intervals and figures",
         ["make_paper_numbers.py", "--cache", CACHE, "--runs", RUNS, "--cost", "model_cost.json", "--paper", PAPER])
    bundle()


def check():
    """Same as selftest.sh: every script on tiny fake data. The numbers it prints mean nothing."""
    T = os.path.join(tempfile.gettempdir(), "suras_selftest")
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    tiny = ["--epochs1", "4", "--epochs2", "2", "--lr2", "1e-4", "--backbone", "tiny", "--batch", "16"]
    step("DBBD unit test", ["dbbd.py"])
    step("fake data", ["make_synthetic_data.py", "--out", os.path.join(T, "fake"), "--n", "24"])
    step("prepare", ["prepare_data.py", "--bdff_dir", os.path.join(T, "fake", "BD-Freshwater-Fish"),
                     "--matsya_dir", os.path.join(T, "fake", "MatsyaDx-BD"), "--out", os.path.join(T, "cache")])
    for b in (1, 2, 4):
        r = os.path.join(T, "runs", f"b{b}")
        step(f"train b={b}", ["train.py", "--cache", os.path.join(T, "cache"), "--b", str(b), "--out", r, *tiny])
        step(f"evaluate b={b}", ["evaluate.py", "--cache", os.path.join(T, "cache"), "--run", r])
        step(f"tflite b={b}", ["export_tflite.py", "--cache", os.path.join(T, "cache"), "--run", r, "--n_rep", "20"])
    step("folds", ["make_folds.py", "--cache", os.path.join(T, "cache"), "--k", "5"])
    for b in (1, 4):
        r = os.path.join(T, "runs", "cv", f"b{b}_f0")
        step(f"cv fold 0 b={b}", ["train.py", "--cache", os.path.join(T, "cache"), "--b", str(b), "--folds",
                                  os.path.join(T, "cache", "folds.json"), "--fold", "0", "--out", r, *tiny])
        step(f"cv evaluate b={b}", ["evaluate.py", "--cache", os.path.join(T, "cache"), "--run", r])
    r = os.path.join(T, "runs", "b4_up")
    step("enlarged control", ["train.py", "--cache", os.path.join(T, "cache"), "--b", "4", "--upsample", "--out", r, *tiny])
    step("enlarged evaluate", ["evaluate.py", "--cache", os.path.join(T, "cache"), "--run", r])
    r = os.path.join(T, "runs", "b2_lodo_EUS")
    step("train LODO", ["train.py", "--cache", os.path.join(T, "cache"), "--b", "2", "--holdout", "EUS", "--out", r, *tiny])
    step("evaluate LODO", ["evaluate.py", "--cache", os.path.join(T, "cache"), "--run", r])
    os.makedirs(os.path.join(T, "paper"), exist_ok=True)
    step("model cost", ["model_cost.py", "--cache", os.path.join(T, "cache"), "--out", os.path.join(T, "model_cost.json")])
    step("paper numbers", ["make_paper_numbers.py", "--cache", os.path.join(T, "cache"), "--runs", os.path.join(T, "runs"),
                           "--cost", os.path.join(T, "model_cost.json"), "--paper", os.path.join(T, "paper")])
    v = subprocess.run([sys.executable, "-c", "import tensorflow as tf; print(tf.__version__)"], env=env(),
                       capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    print(f"\nCHECK PASSED (TensorFlow {''.join(v)}, Python {sys.version.split()[0]}). "
          f"Now set MODE = \"full\" and run again.")


if __name__ == "__main__":
    {"check": check, "full": full, "reeval": reeval, "extra": extra, "upcv": upcv}[MODE]()
