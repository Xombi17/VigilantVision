"""
VigilantVision — Heuristic Threshold Analysis
==============================================
Tests simple rule-based heuristics on features.csv with proper video-level
cross-validation, so results are directly comparable to the logistic
regression classifier.

Usage:
    python3 scripts/heuristic_analysis.py --input_csv features.csv
"""

import csv
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SENTINEL = 1_000_000.0


def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by video
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_name"]].append(r)
    videos = list(by_video.keys())

    X_all, y_all, video_groups = [], [], []
    for vname in videos:
        vrows = by_video[vname]
        for r in vrows:
            wtt = float(r["wrist_to_torso_norm"])
            wti = float(r["wrist_to_item_norm"])
            ni = int(r["num_items_nearby"])
            if abs(wti - SENTINEL) < 1.0:
                wti = float("nan")
            X_all.append([wtt, wti, ni])
            y_all.append(int(r["label"]))
            video_groups.append(vname)

    X = np.array(X_all, dtype=np.float64)
    y = np.array(y_all)

    # Impute NaN globally (acceptable for heuristic baseline comparison)
    col_mean = np.nanmean(X[:, 1])
    X[:, 1] = np.nan_to_num(X[:, 1], nan=col_mean)

    return X, y, video_groups, videos


def get_folds(video_groups, unique_videos, n_splits=5):
    """Same video-level CV split as the classifier."""
    video_to_label = {}
    for vname in unique_videos:
        mask = [vg == vname for vg in video_groups]
        video_to_label[vname] = 1 if y[mask].sum() > 0 else 0

    pos_videos = [v for v in unique_videos if video_to_label[v] == 1]
    neg_videos = [v for v in unique_videos if video_to_label[v] == 0]

    n_pos_val = max(1, int(len(pos_videos) / n_splits))
    n_neg_val = max(1, int(len(neg_videos) / n_splits))

    folds = []
    for i in range(n_splits):
        val_pos = set(pos_videos[i * n_pos_val : (i + 1) * n_pos_val])
        val_neg = set(neg_videos[i * n_neg_val : (i + 1) * n_neg_val])
        val_videos = val_pos | val_neg
        train_mask = np.array([vg not in val_videos for vg in video_groups])
        val_mask = np.array([vg in val_videos for vg in video_groups])
        folds.append((train_mask, val_mask))
    return folds


def run_heuristic(X_val, y_val, name, predict_fn):
    """Run a heuristic and return metrics."""
    y_pred = predict_fn(X_val)
    y_prob = None
    try:
        roc_auc = roc_auc_score(y_val, y_pred)
    except ValueError:
        roc_auc = 0.0

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn = ((y_pred == 0) & (y_val == 0)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()
    tp = ((y_pred == 1) & (y_val == 1)).sum()

    return {
        "name": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="features.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    global y  # needed by get_folds closure
    X, y, video_groups, unique_videos = load_data(args.input_csv)
    folds = get_folds(video_groups, unique_videos, args.n_splits)

    # Define heuristics to test
    heuristics = []

    # H1: wrist close to torso
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        heuristics.append((f"wtt < {thr}", lambda X, t=thr: (X[:, 0] < t).astype(int)))

    # H2: items nearby
    for thr in [1, 2, 3, 5]:
        heuristics.append((f"ni >= {thr}", lambda X, t=thr: (X[:, 2] >= t).astype(int)))

    # H3: combined — wrist close to torso AND items nearby
    for thr in [0.1, 0.15, 0.2, 0.25]:
        heuristics.append(
            (
                f"wtt < {thr} AND ni > 0",
                lambda X, t=thr: ((X[:, 0] < t) & (X[:, 2] > 0)).astype(int),
            )
        )

    # H4: combined — wrist close to torso OR items nearby
    for thr in [0.1, 0.15, 0.2]:
        heuristics.append(
            (
                f"wtt < {thr} OR ni > 0",
                lambda X, t=thr: ((X[:, 0] < t) | (X[:, 2] > 0)).astype(int),
            )
        )

    # H5: wrist close to item (small distance = near an item)
    for thr in [0.5, 1.0, 1.5]:
        heuristics.append((f"wti < {thr}", lambda X, t=thr: (X[:, 1] < t).astype(int)))

    # H6: all three (wtt small, wti small, items nearby)
    heuristics.append(
        (
            "wtt < 0.15 AND wti < 1.0 AND ni > 0",
            lambda X: ((X[:, 0] < 0.15) & (X[:, 1] < 1.0) & (X[:, 2] > 0)).astype(int),
        )
    )

    # Run all heuristics across all folds
    results = {}
    for name, fn in heuristics:
        results[name] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    print(f"{'=' * 80}")
    print(f"  HEURISTIC THRESHOLD ANALYSIS — {args.n_splits}-fold Video-Level CV")
    print(f"  Data: {len(X)} rows, {len(unique_videos)} videos")
    print(f"{'=' * 80}")

    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        X_val = X[val_mask]
        y_val = y[val_mask]

        for name, fn in heuristics:
            m = run_heuristic(X_val, y_val, name, fn)
            results[name]["precision"].append(m["precision"])
            results[name]["recall"].append(m["recall"])
            results[name]["f1"].append(m["f1"])
            results[name]["roc_auc"].append(m["roc_auc"])
            results[name]["tp"] += m["tp"]
            results[name]["fp"] += m["fp"]
            results[name]["fn"] += m["fn"]
            results[name]["tn"] += m["tn"]

    # Sort by average F1 descending and print
    sorted_heuristics = sorted(
        heuristics, key=lambda h: np.mean(results[h[0]]["f1"]), reverse=True
    )

    print(
        f"\n{'Heuristic':<40s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}  {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s}"
    )
    print(f"{'-' * 90}")
    for name, _ in sorted_heuristics:
        r = results[name]
        p_mean = np.mean(r["precision"])
        r_mean = np.mean(r["recall"])
        f_mean = np.mean(r["f1"])
        a_mean = np.mean(r["roc_auc"])
        print(
            f"{name:<40s} {p_mean:>6.3f} {r_mean:>6.3f} {f_mean:>6.3f} {a_mean:>6.3f}  "
            f"{r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d} {r['tn']:>4d}"
        )

    # Best 3
    print(f"\n{'=' * 80}")
    print(f"  TOP 3 HEURISTICS (by F1)")
    print(f"{'=' * 80}")
    for name, _ in sorted_heuristics[:3]:
        r = results[name]
        p_mean = np.mean(r["precision"])
        r_mean = np.mean(r["recall"])
        f_mean = np.mean(r["f1"])
        a_mean = np.mean(r["roc_auc"])
        print(f"\n  {'─' * 50}")
        print(f"  {name}")
        print(f"  {'─' * 50}")
        print(
            f"  Precision: {p_mean:.4f}  |  Recall: {r_mean:.4f}  |  F1: {f_mean:.4f}  |  AUC: {a_mean:.4f}"
        )
        print(f"  Confusion: TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")

    # Compare to logistic regression results
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON WITH LOGISTIC REGRESSION")
    print(f"{'=' * 80}")
    print(f"                        {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s}")
    print(f"  {'─' * 40}")
    print(f"  Logistic Regression    0.013  0.198  0.024  0.393")
    best = sorted_heuristics[0]
    r = results[best[0]]
    print(
        f"  Best heuristic ({best[0]:<20s}  {np.mean(r['precision']):>6.3f} {np.mean(r['recall']):>6.3f} "
        f"{np.mean(r['f1']):>6.3f} {np.mean(r['roc_auc']):>6.3f}"
    )


if __name__ == "__main__":
    main()
