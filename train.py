"""
Step 2 -- train the multi-task compressed-domain classifier (species head + health head)
on DBBD images with block size b.

  python train.py --cache data_cache --b 2 --out runs/b2
  python train.py --cache data_cache --b 2 --holdout EUS --out runs/b2_lodo_EUS   # leave-one-disease-out

Design (matches the paper):
  * input  : DBBD image (224/b x 224/b x 3, values 0..255) -- the same tensor the Pi produces
  * network: MobileNetV2 (ImageNet init) -> GAP -> dropout -> two softmax heads
  * loss   : L = CE_species * m_s + CE_health * m_h, where m_s / m_h mask out samples that have no
             label for that head (BD-Freshwater-Fish has no health label), times inverse-frequency
             class weights
  * schedule: phase 1 frozen backbone (lr 1e-3), phase 2 fine-tune top layers (lr 1e-5) --
             the same two-phase recipe as the existing 9-species classifier
  * augmentation is applied to the 224x224 RGB image BEFORE DBBD, i.e. in the scene domain.
"""
import argparse, csv, json, os, time

import numpy as np
import tensorflow as tf

from dbbd import dbbd_tf

AUTOTUNE = tf.data.AUTOTUNE


def load_cache(cache):
    imgs = np.load(os.path.join(cache, "images_224.npy"), mmap_mode="r")
    rows = list(csv.DictReader(open(os.path.join(cache, "manifest.csv"), encoding="utf-8")))
    vocab = json.load(open(os.path.join(cache, "vocab.json"), encoding="utf-8"))
    sp = np.array([int(r["species"]) for r in rows], dtype=np.int32)
    he = np.array([int(r["health"]) for r in rows], dtype=np.int32)
    split = np.array([r["split"] for r in rows])
    return imgs, rows, vocab, sp, he, split


def fold_split(rows, folds_path, k):
    """Cross-validation split: test = fold k, validation = fold k+1, training = the rest (make_folds.py)."""
    f = json.load(open(folds_path, encoding="utf-8"))
    K, fold_of = f["k"], f["fold_of_group"]
    return np.array(["test" if fold_of[r["group"]] == k else "val" if fold_of[r["group"]] == (k + 1) % K
                     else "train" for r in rows])


def enlarge(x, b):
    """Control: repeat every DBBD value b x b times, back to 224 x 224. Adds no information; only the
    network sees the same input size as at b = 1."""
    return tf.repeat(tf.repeat(x, b, axis=0), b, axis=1)


def select(split, he, which, holdout_id=None):
    m = split == which
    if holdout_id is not None and which in ("train", "val"):
        m &= he != holdout_id          # the held-out disease is never seen in training/validation
    return np.nonzero(m)[0]


def class_weights(labels, n_classes):
    lab = labels[labels >= 0]
    cnt = np.bincount(lab, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes)
    present = cnt > 0
    w[present] = len(lab) / (present.sum() * cnt[present])
    return w.astype(np.float32)


def build_aug():
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomFlip("horizontal"),
        L.RandomRotation(0.04, fill_mode="reflect"),
        L.RandomZoom((-0.10, 0.10), fill_mode="reflect"),
        L.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
    ], name="scene_aug")


def make_dataset(imgs, idx, sp, he, wsp, whe, b, train, batch, seed, up=False):
    sp_t, he_t = tf.constant(sp), tf.constant(he)
    wsp_t, whe_t = tf.constant(wsp), tf.constant(whe)
    aug = build_aug() if train else None

    def fetch(i):
        return np.asarray(imgs[int(i)], dtype=np.uint8)

    ds = tf.data.Dataset.from_tensor_slices(idx.astype(np.int64))
    if train:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)

    def load(i):
        x = tf.numpy_function(fetch, [i], tf.uint8)
        x.set_shape((224, 224, 3))
        return tf.cast(x, tf.float32), i

    def to_example(x, i):
        if train:
            x = aug(x[None], training=True)[0]
            x = tf.image.random_brightness(x, 0.15 * 255.0)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            x = tf.clip_by_value(x, 0.0, 255.0)
        x = dbbd_tf(x, b)                                   # the compressed-domain input
        if up and b > 1:
            x = enlarge(x, b)
        ys, yh = tf.gather(sp_t, i), tf.gather(he_t, i)
        ms = tf.cast(ys >= 0, tf.float32) * tf.gather(wsp_t, tf.maximum(ys, 0))
        mh = tf.cast(yh >= 0, tf.float32) * tf.gather(whe_t, tf.maximum(yh, 0))
        y = {"species": tf.maximum(ys, 0), "health": tf.maximum(yh, 0)}
        w = {"species": ms, "health": mh}
        return x, y, w

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).map(to_example, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def tiny_backbone(side):
    """Small CNN used ONLY by the self-test (no ImageNet weights needed). Not used in the paper."""
    L = tf.keras.layers
    inp = L.Input((side, side, 3))
    x = inp
    for f in (16, 32, 64):
        x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
    return tf.keras.Model(inp, x, name="tiny_backbone")


def build_model(side, n_species, n_health, weights="imagenet", dropout=0.3, backbone="mobilenetv2"):
    L = tf.keras.layers
    inp = L.Input((side, side, 3), name="dbbd_image")           # 0..255, like the Pi
    x = L.Rescaling(1.0 / 127.5, offset=-1.0, name="to_mnv2_range")(inp)
    if backbone == "tiny":
        base = tiny_backbone(side)
        x = base(x)
    else:
        base = tf.keras.applications.MobileNetV2(input_shape=(side, side, 3), include_top=False,
                                                 weights=None if weights == "none" else weights)
        base.trainable = False
        x = base(x, training=False)                              # BN stays in inference mode
    feat = L.GlobalAveragePooling2D(name="feat")(x)
    h = L.Dropout(dropout)(feat)
    out_s = L.Dense(n_species, activation="softmax", name="species")(h)
    out_h = L.Dense(n_health, activation="softmax", name="health")(h)
    return tf.keras.Model(inp, {"species": out_s, "health": out_h}, name="dbbd_multitask"), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"species": tf.keras.losses.SparseCategoricalCrossentropy(),
              "health": tf.keras.losses.SparseCategoricalCrossentropy()},
        weighted_metrics={"species": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
                          "health": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--b", type=int, default=2, choices=[1, 2, 4, 8], help="DBBD block size")
    ap.add_argument("--holdout", default=None, help="health class to hold out (e.g. EUS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs1", type=int, default=15)
    ap.add_argument("--epochs2", type=int, default=20)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--fine_tune_at", type=int, default=100, help="unfreeze MobileNetV2 layers from this index")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "tiny"],
                    help="'tiny' is only for the self-test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--folds", default=None, help="folds.json from make_folds.py (cross-validation)")
    ap.add_argument("--fold", type=int, default=0, help="test fold when --folds is given")
    ap.add_argument("--upsample", action="store_true", help="control: enlarge the DBBD image back to 224x224")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    imgs, rows, vocab, sp, he, split = load_cache(a.cache)
    if a.folds:
        split = fold_split(rows, a.folds, a.fold)
    nS, nH = len(vocab["species"]), len(vocab["health"])
    hold_id = vocab["health"].index(a.holdout) if a.holdout else None

    tr, va = select(split, he, "train", hold_id), select(split, he, "val", hold_id)
    wsp = class_weights(sp[tr], nS)
    whe = class_weights(he[tr], nH)
    print(f"train {len(tr)}  val {len(va)}  | b={a.b} side={224 // a.b} | holdout={a.holdout}")
    print("species weights", np.round(wsp, 2), "\nhealth weights ", np.round(whe, 2))

    ds_tr = make_dataset(imgs, tr, sp, he, wsp, whe, a.b, True, a.batch, a.seed, a.upsample)
    ds_va = make_dataset(imgs, va, sp, he, wsp, whe, a.b, False, a.batch, a.seed, a.upsample)

    side = 224 if a.upsample else 224 // a.b
    model, base = build_model(side, nS, nH, a.weights, backbone=a.backbone)
    ckpt = os.path.join(a.out, "best.keras")
    cbs = lambda: [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(a.out, "log.csv"), append=True),
    ]
    t0 = time.time()
    compile_model(model, a.lr1)
    h1 = model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs1, callbacks=cbs(), verbose=2)

    base.trainable = True
    if a.backbone == "mobilenetv2":
        for layer in base.layers[:a.fine_tune_at]:
            layer.trainable = False
    compile_model(model, a.lr2)
    h2 = model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs1 + a.epochs2,
                   initial_epoch=len(h1.history["loss"]), callbacks=cbs(), verbose=2)
    model.save(ckpt)  # best weights were restored by EarlyStopping

    hist = {k: [float(v) for v in h1.history.get(k, []) + h2.history.get(k, [])] for k in h1.history}
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w", encoding="utf-8"))
    json.dump({"b": a.b, "side": side, "holdout": a.holdout, "seed": a.seed, "vocab": vocab,
               "n_train": int(len(tr)), "n_val": int(len(va)),
               "phase1_epochs": len(h1.history["loss"]), "phase2_epochs": len(h2.history["loss"]),
               "params": int(model.count_params()), "train_minutes": (time.time() - t0) / 60,
               "species_class_weights": wsp.tolist(), "health_class_weights": whe.tolist(),
               "args": vars(a)}, open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), indent=2)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {ckpt}")


if __name__ == "__main__":
    main()
