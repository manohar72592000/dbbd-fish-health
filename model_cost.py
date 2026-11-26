"""
Software-only model cost: parameters and floating-point operations of the multi-task
classifier for each DBBD block size.  Needs no data and no training (cost depends only on
the architecture and input size), so it can be run at any time.

  python model_cost.py [--cache data_cache] [--out model_cost.json]

FLOPs are counted by the TensorFlow profiler on the frozen inference graph
(one multiply-add = 2 FLOPs) for ONE fish crop.
"""
import argparse, json, os

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from train import build_model


def count_flops(model, side):
    fn = tf.function(lambda x: model(x, training=False))
    cf = fn.get_concrete_function(tf.TensorSpec([1, side, side, 3], tf.float32))
    frozen = convert_variables_to_constants_v2(cf)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    prof = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(),
                                         cmd="op", options=opts)
    return int(prof.total_float_ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--out", default="model_cost.json")
    a = ap.parse_args()
    vp = os.path.join(a.cache, "vocab.json")
    if os.path.exists(vp):
        v = json.load(open(vp, encoding="utf-8")); nS, nH = len(v["species"]), len(v["health"])
    else:
        nS, nH = 12, 4          # BD-Freshwater-Fish species, MatsyaDx health classes
    out = {"n_species": nS, "n_health": nH}
    for b in (1, 2, 4):
        side = 224 // b
        model, _ = build_model(side, nS, nH, weights="none")
        out[f"b{b}"] = {"side": side, "params": int(model.count_params()),
                        "flops": count_flops(model, side)}
        print(f"b={b}: input {side}x{side}, {out[f'b{b}']['params']:,} parameters, "
              f"{out[f'b{b}']['flops'] / 1e6:.1f} MFLOPs per crop")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
