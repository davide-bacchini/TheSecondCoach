"""
train_model.py
==============
Phase 3: Train XGBoost regressor on ml_training.csv to predict realized_delta_xT.

Uses GroupKFold cross-validation (leave-one-match-out) for honest evaluation.
Outputs: trained model, feature importance, CV results.

Usage:
  source .venv/bin/activate
  python train_model.py --training wc2022/ml_training.csv --out-dir wc2022
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

# ── Feature configuration ────────────────────────────────────────────────────
FEATURES = [
    # Actor / State
    "start_x", "start_y", "xT_origin", "turnover_danger",
    "dist_to_goal", "angle_to_goal_rad",
    "under_pressure", "score_diff",
    # 360 context (from Pillar 2)
    "n_opponents_5m", "n_opponents_10m", "nearest_opponent_dist",
    "n_teammates_visible", "n_opponents_visible",
    "n_open_teammates", "n_clear_passing_lanes",
    "has_360_data",
    # Action geometry
    "target_x", "target_y", "target_xT",
    "dx", "dy", "action_distance", "action_angle_rad",
    # One-hot
    "act_pass", "act_carry", "act_shot", "act_dispossessed",
    "role_DEF", "role_MID", "role_ATT", "role_GK",
]

TARGET = "realized_delta_xT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", type=str, default="wc2022/ml_training.csv")
    ap.add_argument("--out-dir", type=str, default="wc2022")
    ap.add_argument("--rf", action="store_true", help="Use RandomForest instead of XGBoost")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading training data ...")
    df = pd.read_csv(args.training, low_memory=False)
    print(f"  {len(df):,} rows, {df['match_id'].nunique()} matches")

    # Verify features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURES].fillna(0).copy()
    y = df[TARGET].copy()
    groups = df["match_id"]

    print(f"  Features: {len(FEATURES)}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")

    # ── Build model ──────────────────────────────────────────────────────────
    USE_XGBOOST = not args.rf
    if USE_XGBOOST:
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=15,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
            model_name = "XGBoost"
        except ImportError:
            print("xgboost not installed — falling back to RandomForest")
            USE_XGBOOST = False

    if not USE_XGBOOST:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        model_name = "RandomForest"

    print(f"\nModel: {model_name}")

    # ── Cross-validation (leave-one-match-out) ───────────────────────────────
    n_folds = min(10, len(groups.unique()))
    print(f"\nCross-validation ({n_folds}-fold GroupKFold) ...")
    gkf = GroupKFold(n_splits=n_folds)
    maes, r2s, rank_accs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        maes.append(mae)
        r2s.append(r2)

        # Rank accuracy: for each event, does the model rank the actual
        # best action highest? (Only meaningful if we had candidates per event,
        # here we check if direction of prediction is correct)
        correct_direction = ((preds > 0) == (y_te > 0)).mean()

        test_matches = groups.iloc[test_idx].unique()
        print(f"  Fold {fold+1}: MAE={mae:.5f}  R²={r2:.3f}  "
              f"Dir.Acc={correct_direction:.1%}  "
              f"matches={list(test_matches[:3])}...")

    print(f"\n  ── CV Summary ──")
    print(f"  Mean MAE:  {np.mean(maes):.5f} ± {np.std(maes):.5f}")
    print(f"  Mean R²:   {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    # ── Save CV results ──────────────────────────────────────────────────────
    cv_df = pd.DataFrame({
        "fold": range(1, n_folds + 1),
        "mae": maes,
        "r2": r2s,
    })
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    # ── Train final model on ALL data ────────────────────────────────────────
    print(f"\nTraining final model on all {len(X):,} rows ...")
    model.fit(X, y)

    # Save model
    if USE_XGBOOST:
        model.save_model(str(model_dir / "vaep_xgb.json"))
        print(f"  Saved → {model_dir}/vaep_xgb.json")
    else:
        import joblib
        joblib.dump(model, str(model_dir / "vaep_rf.pkl"))
        print(f"  Saved → {model_dir}/vaep_rf.pkl")

    # ── Feature importance ───────────────────────────────────────────────────
    if USE_XGBOOST:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    feat_imp.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\n═══ FEATURE IMPORTANCE (top 15) ═══")
    for _, row in feat_imp.head(15).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['feature']:30s} {row['importance']:.4f}  {bar}")

    # ── Score ALL training data for comparison ────────────────────────────────
    print(f"\nScoring all training data ...")
    df["predicted_delta_xT"] = model.predict(X)

    # Compare with hand-crafted DQ
    if "decision_quality" in df.columns:
        # Correlation between ML prediction and hand-crafted DQ
        corr = df["predicted_delta_xT"].corr(df["decision_quality"])
        print(f"  Correlation(ML prediction, hand-crafted DQ): {corr:.3f}")

    # Save scored data
    df.to_csv(out_dir / "ml_scored.csv", index=False)
    print(f"  Saved → {out_dir}/ml_scored.csv")

    print(f"\n✅ Done. All outputs → {out_dir}/")
    print(f"   - models/vaep_xgb.json  (trained model)")
    print(f"   - feature_importance.csv")
    print(f"   - cv_results.csv")
    print(f"   - ml_scored.csv")


if __name__ == "__main__":
    main()
