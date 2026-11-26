"""
Step 1 -- build the labelled, leakage-safe dataset for the paper.

  python prepare_data.py --bdff_dir  /data/BD-Freshwater-Fish \
                         --matsya_dir /data/MatsyaDx-BD \
                         --out data_cache  [--detector best.pt] [--unknown_dir /data/non_fish]

What it does
  * walks the two public datasets and reads labels from folder names
      BD-Freshwater-Fish (Das et al. 2024): 12 species incl. Katla(Catla), Rohu, Mrigal -> species label only
      MatsyaDx-BD (2026): Grass carp / Rohu / Silver carp x {Healthy, Bacterial gill disease,
                          Bacterial red disease, EUS}      -> species AND health label
    a label that a dataset does not provide is stored as -1 and masked out of the loss.
  * resizes every image to 224x224 RGB exactly like the deployed pipeline (optionally after
    cropping the fish with the deployed DBBD-YOLO detector, so training crops look like
    deployment crops) and stores them in one uint8 array (fast training later).
  * groups images of the SAME physical fish (MatsyaDx specimen folders) and near-duplicate
    photos (64-bit dHash, flip-aware) and splits BY GROUP 70/15/15, stratified by label,
    so no fish appears in both training and test.
Outputs (in --out): images_224.npy, manifest.csv, vocab.json, summary.json
"""
import argparse, csv, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---- label vocabularies (aliases are matched on the lower-cased relative path) ----
# order matters: longer / more specific aliases first
SPECIES_ALIASES = [
    ("Black rohu", ["black rohu", "black roi", "black rui"]),
    ("Catla", ["katla", "catla", "thala"]),
    ("Rohu", ["rohu", "rui", "roi"]),
    ("Mrigal", ["mrigal", "mrigel", "mori", "mrigala"]),
    ("Grass carp", ["grass carp", "grasscarp", "grass_carp"]),
    ("Common carp", ["common carp", "commoncarp", "common_carp"]),
    ("Mirror carp", ["mirror carp", "mirrorcarp", "mirror_carp"]),
    ("Silver carp", ["silver carp", "silvercarp", "silver_carp"]),
    ("Striped catfish", ["striped catfish", "pangas", "pangasius"]),
    ("Nile tilapia", ["nile tilapia", "tilapia", "telapiya"]),
    ("Long-whiskered catfish", ["long whiskered", "long-whiskered", "longwhiskered", "long_whiskered"]),
    ("Freshwater shark", ["freshwater shark", "fresh water shark", "freshwater_shark"]),
]
HEALTH_ALIASES = [
    ("Healthy", ["healthy", "fresh fish", "normal"]),
    ("Bacterial gill disease", ["gill"]),
    ("Bacterial red disease", ["red disease", "red_disease", "bacterial red", "reddisease"]),
    ("EUS", [" eus ", "epizootic", "ulcerative"]),
]


def _norm(path: str) -> str:
    s = path.lower().replace("\\", "/")
    s = re.sub(r"[^a-z0-9/]+", " ", s)
    return " " + s.replace("/", " / ") + " "


def match_label(rel_path: str, table):
    s = _norm(rel_path)
    for name, aliases in table:
        for a in aliases:
            a_n = " " + re.sub(r"[^a-z0-9]+", " ", a.lower()).strip() + " "
            if a_n in s:
                return name
    return None


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append(os.path.join(dp, fn))
    return sorted(out)


# ---- image loading -------------------------------------------------------------
def load_rgb(path, draft=896):
    im = Image.open(path)
    try:
        im.draft("RGB", (draft, draft))  # fast JPEG down-scaled decode
    except Exception:
        pass
    return im.convert("RGB")


def to_224(im: Image.Image) -> np.ndarray:
    # direct (non-aspect-preserving) resize, as in the deployed pipeline
    return np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def dhash256(arr224: np.ndarray):
    """256-bit difference hash of the image and of its mirror image (packed to 32 bytes each)."""
    g = Image.fromarray(arr224).convert("L")
    def h(img):
        a = np.asarray(img.resize((17, 16), Image.BILINEAR), dtype=np.int16)
        return np.packbits((a[:, 1:] > a[:, :-1]).flatten())
    return h(g), h(g.transpose(Image.FLIP_LEFT_RIGHT))


# ---- optional: crop the fish with the deployed DBBD detector ---------------------
class DBBDCropper:
    """Mirrors deployment: 224 RGB -> DBBD(b=2) -> 128 -> YOLO -> box -> crop from the 224 RGB -> 224."""
    def __init__(self, weights, conf=0.25):
        from ultralytics import YOLO
        from dbbd import dbbd_fast
        self.model, self.conf, self.dbbd = YOLO(weights), conf, dbbd_fast

    def __call__(self, x224: np.ndarray):
        d = self.dbbd(x224, 2)
        d128 = np.asarray(Image.fromarray(d).resize((128, 128), Image.BILINEAR))
        r = self.model.predict(d128, imgsz=128, conf=self.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return x224, 0
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = (r.boxes.xyxy[i].cpu().numpy() * (224.0 / 128.0)).round().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(224, x2), min(224, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return x224, 0
        crop = Image.fromarray(x224[y1:y2, x1:x2]).resize((224, 224), Image.BILINEAR)
        return np.asarray(crop, dtype=np.uint8), 1


# ---- grouping + split -------------------------------------------------------------
class UF:
    def __init__(self, n): self.p, self.sz = list(range(n)), [1] * n
    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def u(self, a, b, cap=None):
        a, b = self.f(a), self.f(b)
        if a == b: return False
        if cap is not None and self.sz[a] + self.sz[b] > cap: return False
        self.p[b] = a; self.sz[a] += self.sz[b]; return True


def _popcount_u8(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):                       # numpy >= 2.0
        return np.bitwise_count(x)
    return np.unpackbits(x, axis=-1)


def near_duplicate_groups(hashes, hflip, labels, seed_groups, max_dist=12, cap=30):
    """Union images of the same specimen folder, then near-duplicate photos (256-bit dHash,
    flip-aware, Hamming <= max_dist) that carry the SAME label.  `cap` stops chains of
    merely-similar photos from snowballing into one giant group."""
    n = len(hashes)
    uf, merges = UF(n), 0
    first = {}
    for i, g in enumerate(seed_groups):                     # same specimen folder (never capped)
        if g in first: uf.u(first[g], i)
        else: first[g] = i
    H, F = np.stack(hashes), np.stack(hflip)
    lab = np.array(labels)
    for i in range(n - 1):
        d = np.minimum(_popcount_u8(H[i] ^ H[i + 1:]).sum(-1), _popcount_u8(H[i] ^ F[i + 1:]).sum(-1))
        for j in np.nonzero((d <= max_dist) & (lab[i + 1:] == lab[i]))[0]:
            merges += uf.u(i, i + 1 + int(j), cap)
    return [uf.f(i) for i in range(n)], merges


def group_split(keys, groups, rng, fracs=(0.70, 0.15, 0.15)):
    """Assign WHOLE groups to train/val/test, stratified by each group's majority label.
    Within a stratum the shuffled groups fill test first, then val (each gets >= 1 group when
    the stratum has >= 3 groups), and the rest go to train."""
    members = defaultdict(list)
    for i, g in enumerate(groups): members[g].append(i)
    strata = defaultdict(list)
    for g, idx in members.items():
        strata[Counter(keys[i] for i in idx).most_common(1)[0][0]].append(g)
    split = [None] * len(keys)
    for _, gs in sorted(strata.items()):
        gs = sorted(gs); rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        need = {"test": fracs[2] * total, "val": fracs[1] * total}
        got = {"test": 0, "val": 0}
        for k, g in enumerate(gs):
            remaining = len(gs) - k
            if len(gs) >= 3 and (got["test"] == 0 or got["test"] + len(members[g]) / 2 <= need["test"]) and remaining > 2:
                s = "test"
            elif len(gs) >= 3 and (got["val"] == 0 or got["val"] + len(members[g]) / 2 <= need["val"]) and remaining > 1:
                s = "val"
            else:
                s = "train"
            if s != "train": got[s] += len(members[g])
            for i in members[g]: split[i] = s
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdff_dir", help="BD-Freshwater-Fish root (species labels)")
    ap.add_argument("--matsya_dir", help="MatsyaDx-BD root (species + health labels)")
    ap.add_argument("--unknown_dir", help="optional: non-fish / background crops -> species 'Unknown'")
    ap.add_argument("--out", default="data_cache")
    ap.add_argument("--detector", help="optional DBBD-YOLO weights (.pt) to crop fish like deployment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_class", type=int, default=0, help="debug: cap images per class")
    ap.add_argument("--dup_dist", type=int, default=12, help="max Hamming distance (of 256) for near-duplicates")
    ap.add_argument("--max_group", type=int, default=30, help="largest allowed near-duplicate group")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    species_names = [s for s, _ in SPECIES_ALIASES]
    health_names = [h for h, _ in HEALTH_ALIASES]
    if a.unknown_dir: species_names.append("Unknown")

    items = []  # (path, source, species_name|None, health_name|None, seed_group)
    def add(root, source, want_health):
        per_class = Counter()
        for p in list_images(root):
            rel = os.path.relpath(p, root)
            sp = "Unknown" if source == "unknown" else match_label(rel, SPECIES_ALIASES)
            he = match_label(rel, HEALTH_ALIASES) if want_health else None
            if source == "matsya" and he is None:
                print(f"[skip] no health label in path: {rel}", file=sys.stderr); continue
            if source == "bdff" and sp is None:
                print(f"[skip] no species label in path: {rel}", file=sys.stderr); continue
            k = (sp, he)
            if a.max_per_class and per_class[k] >= a.max_per_class: continue
            per_class[k] += 1
            parent = os.path.basename(os.path.dirname(p))
            is_class_folder = (match_label(parent, SPECIES_ALIASES) or match_label(parent, HEALTH_ALIASES))
            seed = f"{source}:{os.path.dirname(rel)}" if (source == "matsya" and not is_class_folder) else f"{source}:{rel}"
            items.append((p, source, sp, he, seed))
    if a.bdff_dir: add(a.bdff_dir, "bdff", False)
    if a.matsya_dir: add(a.matsya_dir, "matsya", True)
    if a.unknown_dir: add(a.unknown_dir, "unknown", False)
    if not items: sys.exit("no images found -- check the dataset paths")
    print(f"{len(items)} images found")

    cropper = DBBDCropper(a.detector) if a.detector else None
    N = len(items)
    arr = np.lib.format.open_memmap(os.path.join(a.out, "images_224.npy"), mode="w+",
                                    dtype=np.uint8, shape=(N, 224, 224, 3))
    hashes, hflips, det = [], [], []
    for i, (p, *_rest) in enumerate(items):
        x = to_224(load_rgb(p))
        found = -1
        if cropper is not None:
            x, found = cropper(x)
        arr[i] = x
        h, hf = dhash256(x); hashes.append(h); hflips.append(hf); det.append(found)
        if (i + 1) % 250 == 0: print(f"  cached {i + 1}/{N}")
    arr.flush()

    keys = [f"{it[1]}|{it[2]}|{it[3]}" for it in items]
    groups, n_merges = [], 0
    for src in sorted({it[1] for it in items}):
        idx = [i for i, it in enumerate(items) if it[1] == src]
        g, mrg = near_duplicate_groups([hashes[i] for i in idx], [hflips[i] for i in idx],
                                       [keys[i] for i in idx], [items[i][4] for i in idx],
                                       a.dup_dist, a.max_group)
        n_merges += mrg
        groups.extend([(i, f"{src}:{gi}") for i, gi in zip(idx, g)])
    groups = [g for _, g in sorted(groups)]
    split = group_split(keys, groups, np.random.default_rng(a.seed))

    sp_idx = {s: k for k, s in enumerate(species_names)}
    he_idx = {h: k for k, h in enumerate(health_names)}
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "path", "source", "species", "health", "group", "split", "det_found"])
        for i, (p, src, sp, he, _) in enumerate(items):
            w.writerow([i, p, src, sp_idx.get(sp, -1), he_idx.get(he, -1), groups[i], split[i], det[i]])
    json.dump({"species": species_names, "health": health_names},
              open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8"), indent=2)

    # ---- summary + LaTeX table rows for the paper -----------------------------------
    summ = {"n_images": N, "n_groups": len(set(groups)), "near_duplicate_merges": int(n_merges),
            "largest_group": int(max(Counter(groups).values())),
            "by_source_split": {}, "species": {}, "health": {}}
    for src in sorted({it[1] for it in items}):
        summ["by_source_split"][src] = dict(Counter(split[i] for i in range(N) if items[i][1] == src))
    for s in species_names:
        summ["species"][s] = dict(Counter(split[i] for i in range(N) if items[i][2] == s))
    for h in health_names:
        summ["health"][h] = dict(Counter(split[i] for i in range(N) if items[i][3] == h))
    if cropper is not None:
        summ["detector_found_rate"] = float(np.mean([d == 1 for d in det]))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
