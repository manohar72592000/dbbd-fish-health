"""
Step 3 -- evaluate a trained run on its TEST split and write metrics.json.

  python evaluate.py --cache data_cache --run runs/b2
  python evaluate.py --cache data_cache --run runs/b2_lodo_EUS     # also scores the held-out disease

Reports
  species : accuracy, macro-F1, per-class P/R/F1 (all species-labelled test images), plus accuracy on
            diseased vs healthy MatsyaDx fish (is species ID robust to disease?)
  health  : 4-class accuracy, macro-F1, per-class P/R/F1; binary healthy-vs-diseased sensitivity /
            specificity (argmax of the health head)
  abnormality score (Eq. in paper): squared Mahalanobis distance of the GAP feature to the HEALTHY
            training features (Ledoit-Wolf shrinkage covariance).  Compared against the softmax
            baseline 1 - p(healthy).  AUROC and FPR@95%TPR for (i) seen diseases vs healthy and
            (ii) the held-out disease vs healthy in leave-one-disease-out runs.  The held-out disease
            is removed from training and validation, so ALL its images (every split) are scored against
            the healthy TEST fish; the test-split-only figures are kept under "unseen_test_only".
  decision rule: threshold tau_a = 95th percentile of healthy VALIDATION scores; a fish is reported
            "abnormal" if score > tau_a OR the health head says diseased.  For held-out disease we
            report the fraction NOT reported healthy ("flag rate").
"""
import argparse, json, os

import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from dbbd import dbbd_fast
from train import fold_split, load_cache


def predict(model, feat_model, imgs, idx, b, bs=64, up=False):
    P_s, P_h, F = [], [], []
    for k in range(0, len(idx), bs):
        x = np.stack([dbbd_fast(np.asarray(imgs[i]), b) for i in idx[k:k + bs]]).astype(np.float32)
        if up and b > 1:                                      # control: same values, 224 x 224 grid
            x = x.repeat(b, axis=1).repeat(b, axis=2)
        out = model.predict(x, verbose=0)
        P_s.append(out["species"]); P_h.append(out["health"])
        F.append(feat_model.predict(x, verbose=0))
    return np.concatenate(P_s), np.concatenate(P_h), np.concatenate(F)


def fpr_at_tpr(y, s, tpr=0.95):
    fpr, t, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(t, tpr)]) if (t >= tpr).any() else 1.0


def per_class(y, p, names):
    labels = list(range(len(names)))
    P, R, F, N = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {names[k]: {"precision": float(P[k]), "recall": float(R[k]), "f1": float(F[k]), "n": int(N[k])}
            for k in labels if N[k] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, "config.json"), encoding="utf-8"))
    b, holdout = cfg["b"], cfg["holdout"]
    args = cfg.get("args", {})
    up = bool(args.get("upsample"))
    imgs, rows, vocab, sp, he, split = load_cache(a.cache)
    if args.get("folds"):
        split = fold_split(rows, args["folds"], args["fold"])
    S, Hn = vocab["species"], vocab["health"]
    healthy = Hn.index("Healthy")
    hold_id = Hn.index(holdout) if holdout else None
    src = np.array([r["source"] for r in rows])

    model = tf.keras.models.load_model(os.path.join(a.run, "best.keras"))
    feat_model = tf.keras.Model(model.input, model.get_layer("feat").output)

    te = np.nonzero(split == "test")[0]
    va = np.nonzero((split == "val") & ((he != hold_id) if hold_id is not None else True))[0]
    tr_h = np.nonzero((split == "train") & (he == healthy))[0]

    ps_te, ph_te, f_te = predict(model, feat_model, imgs, te, b, up=up)
    _, ph_va, f_va = predict(model, feat_model, imgs, va, b, up=up)
    _, _, f_trh = predict(model, feat_model, imgs, tr_h, b, up=up)

    M = {"b": b, "side": cfg.get("side", 224 // b), "holdout": holdout, "n_test": int(len(te)),
         "upsample": up, "fold": args.get("fold") if args.get("folds") else None}

    # ---------------- species ----------------
    m = sp[te] >= 0
    if hold_id is not None:
        m &= he[te] != hold_id
    ys, yp = sp[te][m], ps_te[m].argmax(1)
    M["species"] = {"acc": float(accuracy_score(ys, yp)),
                    "macro_f1": float(f1_score(ys, yp, average="macro")),
                    "n": int(m.sum()), "per_class": per_class(ys, yp, S),
                    "confusion": confusion_matrix(ys, yp, labels=range(len(S))).tolist()}
    for name, sel in (("acc_on_diseased_fish", (he[te] >= 0) & (he[te] != healthy)),
                      ("acc_on_healthy_farm_fish", he[te] == healthy)):
        mm = m & sel
        if mm.any():
            M["species"][name] = float(accuracy_score(sp[te][mm], ps_te[mm].argmax(1)))

    # ---------------- health ----------------
    mh = he[te] >= 0
    if hold_id is not None:
        mh &= he[te] != hold_id
    yh, hp = he[te][mh], ph_te[mh].argmax(1)
    M["health"] = {"acc": float(accuracy_score(yh, hp)),
                   "macro_f1": float(f1_score(yh, hp, average="macro")),
                   "n": int(mh.sum()), "per_class": per_class(yh, hp, Hn),
                   "confusion": confusion_matrix(yh, hp, labels=range(len(Hn))).tolist()}
    dis_true, dis_pred = yh != healthy, hp != healthy
    M["health"]["sensitivity"] = float((dis_pred & dis_true).sum() / max(dis_true.sum(), 1))
    M["health"]["specificity"] = float((~dis_pred & ~dis_true).sum() / max((~dis_true).sum(), 1))

    # ---------------- abnormality score ----------------
    lw = LedoitWolf().fit(f_trh)
    mu, prec = lw.location_, lw.precision_
    md = lambda F: np.einsum("ij,jk,ik->i", F - mu, prec, F - mu)
    s_md_te, s_md_va = md(f_te), md(f_va)
    s_msp_te, s_msp_va = 1.0 - ph_te[:, healthy], 1.0 - ph_va[:, healthy]
    tau_md = float(np.percentile(s_md_va[he[va] == healthy], 95))
    tau_msp = float(np.percentile(s_msp_va[he[va] == healthy], 95))
    M["abnormality"] = {"tau_md": tau_md, "tau_msp": tau_msp, "n_healthy_train_for_fit": int(len(tr_h))}

    # further baselines for flagging unseen conditions; higher = more abnormal
    Wh, ch = model.get_layer("health").get_weights()
    unit = lambda F: F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    bank = unit(f_trh)
    kk = min(5, len(bank))
    pca = PCA(n_components=min(16, len(f_trh) - 1)).fit(f_trh)
    lw_p = LedoitWolf().fit(pca.transform(f_trh))
    def extra_scores(F, PH):
        Z = pca.transform(F) - lw_p.location_
        return {"energy": -logsumexp(F @ Wh + ch, axis=1),             # energy score of the health head
                "maxsm": 1.0 - PH.max(1),                              # 1 - max softmax over health classes
                "knn": -np.sort(unit(F) @ bank.T, axis=1)[:, -kk],     # cosine distance to 5th-nearest healthy
                "pcamd": np.einsum("ij,jk,ik->i", Z, lw_p.precision_, Z)}   # Mahalanobis in 16-D PCA space
    x_te = extra_scores(f_te, ph_te)

    def score_block(sel_pos):
        sel = (he[te] == healthy) | sel_pos
        y = sel_pos[sel].astype(int)
        out = {}
        for nm, s in (("md", s_md_te[sel]), ("msp", s_msp_te[sel]), *((k, v[sel]) for k, v in x_te.items())):
            out[f"auroc_{nm}"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
            out[f"fpr95_{nm}"] = fpr_at_tpr(y, s) if 0 < y.sum() < len(y) else None
        return out

    seen = (he[te] >= 0) & (he[te] != healthy)
    if hold_id is not None:
        seen &= he[te] != hold_id
    M["abnormality"]["seen_vs_healthy"] = score_block(seen)
    flag = (s_md_te > tau_md) | (ph_te.argmax(1) != healthy)
    # simple rule: flag when 1 - p_healthy exceeds its own healthy-validation 95th percentile
    flag_msp = (s_msp_te > tau_msp) | (ph_te.argmax(1) != healthy)
    M["abnormality"]["false_alarm_on_healthy_test"] = float(flag[he[te] == healthy].mean())
    M["abnormality"]["false_alarm_msp_rule"] = float(flag_msp[he[te] == healthy].mean())
    if hold_id is not None:
        # test-split-only figures, kept for reference
        unseen = he[te] == hold_id
        M["abnormality"]["unseen_test_only"] = dict(
            score_block(unseen), flag_rate_rule=float(flag[unseen].mean()),
            flag_rate_msp_rule=float(flag_msp[unseen].mean()), n_images=int(unseen.sum()))
        # main figures: the held-out disease was removed from training AND validation, so every one of
        # its images (train/val/test splits alike) is unseen -> score all of them against healthy TEST fish
        ev = np.nonzero(he == hold_id)[0]
        _, ph_u, f_u = predict(model, feat_model, imgs, ev, b, up=up)
        s_md_u, s_msp_u = md(f_u), 1.0 - ph_u[:, healthy]
        x_u = extra_scores(f_u, ph_u)
        h_te = he[te] == healthy
        y = np.r_[np.zeros(h_te.sum(), int), np.ones(len(ev), int)]
        out = {}
        for nm, sh, su in (("md", s_md_te[h_te], s_md_u), ("msp", s_msp_te[h_te], s_msp_u),
                           *((k, x_te[k][h_te], x_u[k]) for k in x_te)):
            s = np.r_[sh, su]
            out[f"auroc_{nm}"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
            out[f"fpr95_{nm}"] = fpr_at_tpr(y, s) if 0 < y.sum() < len(y) else None
        grp = np.array([r.get("group", r["idx"]) for r in rows])
        out["n_images"], out["n_fish"] = int(len(ev)), int(len(set(grp[ev])))
        M["abnormality"]["unseen_vs_healthy"] = out
        flag_u = (s_md_u > tau_md) | (ph_u.argmax(1) != healthy)
        M["abnormality"]["unseen_flag_rate_rule"] = float(flag_u.mean())
        M["abnormality"]["unseen_flag_rate_head_only"] = float((ph_u.argmax(1) != healthy).mean())
        M["abnormality"]["unseen_flag_rate_msp_rule"] = float(((s_msp_u > tau_msp) | (ph_u.argmax(1) != healthy)).mean())

    json.dump(M, open(os.path.join(a.run, "metrics.json"), "w", encoding="utf-8"), indent=2)
    np.savez_compressed(os.path.join(a.run, "test_scores.npz"), idx=te, ps=ps_te, ph=ph_te,
                        s_md=s_md_te, s_msp=s_msp_te)
    sv = M["abnormality"]["seen_vs_healthy"]
    print(f"[{a.run}] species acc {M['species']['acc']:.4f}  health acc {M['health']['acc']:.4f} "
          f"sens {M['health']['sensitivity']:.3f} spec {M['health']['specificity']:.3f}  "
          f"AUROC(MD) seen {sv['auroc_md']}")
    if hold_id is not None:
        print("  unseen", M["abnormality"]["unseen_vs_healthy"], "flag rate", M["abnormality"]["unseen_flag_rate_rule"])


if __name__ == "__main__":
    main()
