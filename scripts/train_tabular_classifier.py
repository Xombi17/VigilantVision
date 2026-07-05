"""
VigilantVision — Tabular Classifier Training
=============================================
Trains a logistic regression classifier on the geometric features
from features.csv, with proper video-level cross-validation.

Handles:
  - 1e6 sentinel values in wrist_to_item_norm (replaced with NaN, mean-imputed)
  - Extreme class imbalance via class_weight='balanced'
  - Video-level train/eval split (never split frames from same video)
  - Reports precision, recall, F1, ROC-AUC honestly

Usage:
    python3 scripts/train_tabular_classifier.py
            --input_csv features.csv
"""

import csv
import argparse
import warnings
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")  # keep output clean

SENTINEL = 1_000_000.0


def load_and_clean(path):
    """Load features.csv, fix sentinel values, return arrays + video groups."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} rows from {path}")

    # Group by video for video-level splitting
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_name"]].append(r)

    videos = list(by_video.keys())
    print(f"Unique videos: {len(videos)}")

    # Build feature matrix
    X_all, y_all, video_groups = [], [], []
    for vname in videos:
        vrows = by_video[vname]
        for r in vrows:
            wtt = float(r["wrist_to_torso_norm"])
            wti = float(r["wrist_to_item_norm"])
            ni  = int(r["num_items_nearby"])

            # Fix sentinel: replace 1e6 with NaN
            if abs(wti - SENTINEL) < 1.0:
                wti = float("nan")

            X_all.append([wtt, wti, ni])
            y_all.append(int(r["label"]))
            video_groups.append(vname)

    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all)

    # Count positives
    total_pos = y_all.sum()
    print(f"Positive frames: {total_pos} ({total_pos/len(y_all)*100:.2f}%)")
    print(f"Negative frames: {len(y_all) - total_pos}")

    # Count NaN in wrist_to_item_norm
    nan_count = np.isnan(X_all[:, 1]).sum()
    print(f"Rows with sentinel (NaN after fix): {nan_count} ({nan_count/len(X_all)*100:.1f}%)")
    print(f"NaN imputation will be done per-fold during CV (to avoid data leakage)")

    return X_all, y_all, video_groups, videos


def video_level_cv(X, y, video_groups, unique_videos, n_splits=5):
    """
    Stratified video-level cross-validation.
    Splits by unique video, stratified by whether a video has any positive frames.
    """
    # Determine each video's label (1 if any positive frame)
    video_to_label = {}
    for vname in unique_videos:
        mask = [vg == vname for vg in video_groups]
        video_to_label[vname] = 1 if y[mask].sum() > 0 else 0

    pos_videos = [v for v in unique_videos if video_to_label[v] == 1]
    neg_videos = [v for v in unique_videos if video_to_label[v] == 0]

    print(f"\nVideo-level split: {len(pos_videos)} positive videos, {len(neg_videos)} negative videos")

    np.random.shuffle(pos_videos)
    np.random.shuffle(neg_videos)

    # Per-fold: hold out ~20% of pos and ~20% of neg videos
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


def evaluate(y_true, y_pred, y_prob, split_name="val"):
    """Print and return metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} METRICS")
    print(f"{'='*50}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  Confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*50}\n")

    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="features.csv")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength (smaller = stronger reg)")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    X, y, video_groups, unique_videos = load_and_clean(args.input_csv)

    # Scale features
    scaler = StandardScaler()

    # Video-level CV
    folds = video_level_cv(X, y, video_groups, unique_videos, n_splits=args.n_splits)

    all_metrics = defaultdict(list)
    best_model = None
    best_score = 0

    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        print(f"\n{'#'*50}")
        print(f"  FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'#'*50}")

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        # Count train/val positive frames
        print(f"  Train: {y_train.sum()} pos / {len(y_train)} total "
              f"({y_train.sum()/len(y_train)*100:.2f}%)")
        print(f"  Val:   {y_val.sum()} pos / {len(y_val)} total "
              f"({y_val.sum()/len(y_val)*100:.2f}%)")

        # Impute NaN per-fold (train mean only, to avoid data leakage)
        col_mean_train = np.nanmean(X_train[:, 1])
        X_train[:, 1] = np.nan_to_num(X_train[:, 1], nan=col_mean_train)
        X_val[:, 1] = np.nan_to_num(X_val[:, 1], nan=col_mean_train)

        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train
        model = LogisticRegression(
            C=args.C,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        metrics = evaluate(y_val, y_pred, y_prob, split_name=f"fold {fold_idx + 1}")

        for k, v in metrics.items():
            all_metrics[k].append(v)

        # Save best model by ROC-AUC
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_scaler = scaler

    # Summary across folds
    print(f"\n{'='*50}")
    print(f"  CROSS-VALIDATION SUMMARY ({args.n_splits} folds)")
    print(f"{'='*50}")
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric:12s}:  {mean_val:.4f}  ±  {std_val:.4f}")
    print(f"{'='*50}\n")

    # Save the best model
    import joblib
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_model, "scaler": best_scaler}, "models/tabular_classifier.joblib")
    print(f"Best model (ROC-AUC={best_score:.4f}) saved to models/tabular_classifier.joblib")

    # Print coefficients
    coef_names = ["wrist_to_torso_norm", "wrist_to_item_norm", "num_items_nearby"]
    print(f"\n  Model Coefficients:")
    for name, coef in zip(coef_names, best_model.coef_[0]):
        print(f"    {name:25s}:  {coef:+.4f}")
    print(f"    Intercept:         {best_model.intercept_[0]:+.4f}")
    print()

    # Retrain on full data for final deployment-ready model
    print("Retraining on full dataset for deployment...")

    # Impute NaN using global column mean (acceptable for final model, no leakage concern)
    col_mean_full = np.nanmean(X[:, 1])
    X_full = X.copy()
    X_full[:, 1] = np.nan_to_num(X_full[:, 1], nan=col_mean_full)
    print(f"Imputed NaN with global column mean: {col_mean_full:.4f}")

    X_scaled_full = scaler.fit_transform(X_full)
    final_model = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    final_model.fit(X_scaled_full, y)
    joblib.dump(
        {"model": final_model, "scaler": scaler},
        "models/tabular_classifier_full.joblib",
    )
    print("Full-dataset model saved to models/tabular_classifier_full.joblib")


if __name__ == "__main__":
    main()
