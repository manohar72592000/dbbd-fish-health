"""
Step 4 -- export a trained run to TensorFlow Lite for the Raspberry Pi and check that
quantisation does not change the results.

  python export_tflite.py --cache data_cache --run runs/b2

Writes runs/b2/model_fp16.tflite, runs/b2/model_int8.tflite and runs/b2/tflite.json
(file sizes + test accuracy of each TFLite model).
"""
import argparse, json, os, tempfile

import numpy as np
import tensorflow as tf

from dbbd import dbbd_fast
from train import load_cache


def convert(model, mode, rep_images=None):
    with tempfile.TemporaryDirectory() as d:
        model.export(d)                                    # SavedModel (Keras 3)
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        if mode == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif mode == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            def rep():
                for x in rep_images:
                    yield [x[None].astype(np.float32)]
            conv.representative_dataset = rep              # float in/out, int8 weights+activations
        return conv.convert()


def run_tflite(blob, X):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    res = {o["name"]: [] for o in outs}
    for x in X:
        it.set_tensor(inp["index"], x[None].astype(np.float32))
        it.invoke()
        for o in outs:
            res[o["name"]].append(it.get_tensor(o["index"])[0])
    return {k: np.stack(v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_rep", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=0, help="0 = whole test split")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, "config.json"), encoding="utf-8"))
    b = cfg["b"]
    imgs, rows, vocab, sp, he, split = load_cache(a.cache)
    model = tf.keras.models.load_model(os.path.join(a.run, "best.keras"))
    nS, nH = len(vocab["species"]), len(vocab["health"])

    va = np.nonzero(split == "val")[0][:a.n_rep]
    te = np.nonzero(split == "test")[0]
    if a.n_test: te = te[:a.n_test]
    rep = [dbbd_fast(np.asarray(imgs[i]), b).astype(np.float32) for i in va]
    Xte = np.stack([dbbd_fast(np.asarray(imgs[i]), b).astype(np.float32) for i in te])

    ref = model.predict(Xte, verbose=0)
    out = {"keras": {"species_acc": None, "health_acc": None}}

    def accs(ps, ph):
        ms, mh = sp[te] >= 0, he[te] >= 0
        return {"species_acc": float((ps.argmax(1)[ms] == sp[te][ms]).mean()) if ms.any() else None,
                "health_acc": float((ph.argmax(1)[mh] == he[te][mh]).mean()) if mh.any() else None}
    out["keras"] = accs(ref["species"], ref["health"])

    for mode in ("fp16", "int8"):
        blob = convert(model, mode, rep)
        path = os.path.join(a.run, f"model_{mode}.tflite")
        open(path, "wb").write(blob)
        r = run_tflite(blob, Xte)
        # identify which output is which by its width
        by_w = {v.shape[1]: v for v in r.values()}
        if nS == nH:   # ambiguous widths: fall back to signature order (species, health)
            vals = list(r.values()); ps, ph = vals[0], vals[1]
        else:
            ps, ph = by_w[nS], by_w[nH]
        out[mode] = {"size_mb": len(blob) / 1e6, **accs(ps, ph),
                     "agree_species_with_keras": float((ps.argmax(1) == ref["species"].argmax(1)).mean()),
                     "agree_health_with_keras": float((ph.argmax(1) == ref["health"].argmax(1)).mean())}
        print(mode, out[mode])
    json.dump(out, open(os.path.join(a.run, "tflite.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
