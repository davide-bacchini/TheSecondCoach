"""
build_training_data.py
======================
Phase 2: Transform decision_quality.csv into ML-ready training data.

Uses YOUR compute_metrics.py output (4-pillar architecture) as the base,
adds geometry features from BLACKBOX approach, and computes realized_delta_xT
as the training label.

Usage:
  source .venv/bin/activate
  python build_training_data.py --input wc2022/decision_quality.csv --out-dir wc2022
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ── xT grid lookup (loaded from compute_metrics.py output) ───────────────────
PITCH_LEN, PITCH_WID = 120.0, 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
XT_COLS, XT_ROWS = 16, 12


def load_xt_grid(path: Path) -> np.ndarray:
    """Load xT grid CSV produced by compute_metrics.py."""
    df = pd.read_csv(path, index_col=0)
    return df.values


def xT_lookup(grid, x, y):
    """Look up xT value at pitch coordinates."""
    xi = int(np.clip(x / PITCH_LEN * XT_COLS, 0, XT_COLS - 1))
    yi = int(np.clip(y / PITCH_WID * XT_ROWS, 0, XT_ROWS - 1))
    return float(grid[yi, xi])


def angle_to_goal(x, y, gx=GOAL_X, post1=36.0, post2=44.0):
    """Geometric shooting angle (radians)."""
    a = np.array([gx - x, post1 - y])
    b = np.array([gx - x, post2 - y])
    cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.arccos(np.clip(cos_t, -1, 1)))


def get_role(position_name):
    """Map StatsBomb position name to role category."""
    if pd.isna(position_name):
        return "MID"
    pn = str(position_name).lower()
    if "goalkeeper" in pn or "keeper" in pn:
        return "GK"
    elif any(w in pn for w in ["back", "center back", "wing back"]):
        return "DEF"
    elif any(w in pn for w in ["midfield", "defensive mid", "attacking mid"]):
        return "MID"
    elif any(w in pn for w in ["wing", "forward", "striker", "center forward"]):
        return "ATT"
    return "MID"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="wc2022/decision_quality.csv")
    ap.add_argument("--xt-grid", type=str, default="wc2022/xt_grid.csv")
    ap.add_argument("--out-dir", type=str, default="wc2022")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading decision quality data ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} total rows")

    print("Loading xT grid ...")
    xt_grid = load_xt_grid(args.xt_grid)
    print(f"  Grid shape: {xt_grid.shape}, range: [{xt_grid.min():.6f}, {xt_grid.max():.6f}]")

    # ── Filter to action types with known endpoints ──────────────────────────
    # Training: Pass, Carry, Shot (have real endpoints for label)
    # Also include Dispossessed, Miscontrol (turnover events with no endpoint)
    ACTION_TYPES = ["Pass", "Carry", "Shot", "Dispossessed", "Miscontrol"]
    actions = df[df["event_type"].isin(ACTION_TYPES)].copy()
    print(f"  {len(actions):,} actionable rows (Pass/Carry/Shot/Dispossessed/Miscontrol)")

    # ── Drop rows with missing start coordinates ─────────────────────────────
    before = len(actions)
    actions = actions.dropna(subset=["start_x", "start_y"])
    print(f"  Dropped {before - len(actions)} rows with missing start coordinates")

    # ── Compute realized_delta_xT (the ML training label) ────────────────────
    # Uses the ACTUAL outcome from the data, not hand-crafted formulas
    print("Computing realized_delta_xT labels ...")

    def compute_realized_delta_xt(row):
        sx, sy = row["start_x"], row["start_y"]
        xt_from = xT_lookup(xt_grid, sx, sy)
        etype = row["event_type"]

        if etype == "Pass":
            if row.get("is_turnover"):
                # Incomplete pass: ball lost
                return -xt_from * 0.5
            ex, ey = row.get("end_x"), row.get("end_y")
            if pd.notna(ex) and pd.notna(ey):
                return xT_lookup(xt_grid, ex, ey) - xt_from
            return 0.0

        elif etype == "Carry":
            ex, ey = row.get("end_x"), row.get("end_y")
            if pd.notna(ex) and pd.notna(ey):
                return xT_lookup(xt_grid, ex, ey) - xt_from
            return 0.0

        elif etype == "Shot":
            outcome = str(row.get("outcome", ""))
            if "Goal" in outcome:
                return 1.0 - xt_from  # maximum reward
            elif outcome in ("Saved", "Blocked"):
                return -xt_from * 0.25
            else:
                return -xt_from * 0.4

        elif etype in ("Dispossessed", "Miscontrol"):
            return -xt_from  # total loss of possession

        return 0.0

    actions["realized_delta_xT"] = actions.apply(compute_realized_delta_xt, axis=1)

    # ── Build geometry features ──────────────────────────────────────────────
    print("Building geometry features ...")

    # Target coordinates (where the action went)
    actions["target_x"] = actions.apply(
        lambda r: float(r["end_x"]) if pd.notna(r.get("end_x")) else (
            GOAL_X if r["event_type"] == "Shot" else float(r["start_x"])
        ), axis=1
    )
    actions["target_y"] = actions.apply(
        lambda r: float(r["end_y"]) if pd.notna(r.get("end_y")) else (
            GOAL_Y if r["event_type"] == "Shot" else float(r["start_y"])
        ), axis=1
    )

    # xT at target
    actions["target_xT"] = actions.apply(
        lambda r: xT_lookup(xt_grid, r["target_x"], r["target_y"]), axis=1)

    # xT at origin (from our compute_metrics.py)
    actions["xT_origin"] = actions["xt_from"]

    # Turnover danger: opponent's xT if they get ball here
    actions["turnover_danger"] = actions.apply(
        lambda r: xT_lookup(xt_grid, PITCH_LEN - r["start_x"], r["start_y"]), axis=1)

    # Direction and distance
    actions["dx"] = actions["target_x"] - actions["start_x"]
    actions["dy"] = actions["target_y"] - actions["start_y"]
    actions["action_distance"] = np.sqrt(actions["dx"]**2 + actions["dy"]**2)
    actions["action_angle_rad"] = np.arctan2(actions["dy"], actions["dx"])

    # Distance and angle to goal
    actions["dist_to_goal"] = np.sqrt(
        (GOAL_X - actions["start_x"])**2 + (GOAL_Y - actions["start_y"])**2)
    actions["angle_to_goal_rad"] = actions.apply(
        lambda r: angle_to_goal(r["start_x"], r["start_y"]), axis=1)

    # Has 360 data flag
    actions["has_360_data"] = actions["n_opponents_5m"].notna().astype(int)

    # ── Role encoding ────────────────────────────────────────────────────────
    # We don't have position_name in decision_quality.csv, so we'll use
    # a heuristic based on start_x position in the match
    # If player is in own third on average → DEF, middle → MID, final → ATT
    # This is a rough proxy but works for training
    def position_heuristic(row):
        x = row["start_x"]
        if x < 40:
            return "DEF"
        elif x < 80:
            return "MID"
        else:
            return "ATT"

    actions["role"] = actions.apply(position_heuristic, axis=1)

    # ── One-hot encode ───────────────────────────────────────────────────────
    # Action type
    actions["act_pass"] = (actions["event_type"] == "Pass").astype(int)
    actions["act_carry"] = (actions["event_type"] == "Carry").astype(int)
    actions["act_shot"] = (actions["event_type"] == "Shot").astype(int)
    actions["act_dispossessed"] = (actions["event_type"].isin(
        ["Dispossessed", "Miscontrol"])).astype(int)

    # Role
    actions["role_DEF"] = (actions["role"] == "DEF").astype(int)
    actions["role_MID"] = (actions["role"] == "MID").astype(int)
    actions["role_ATT"] = (actions["role"] == "ATT").astype(int)
    actions["role_GK"] = 0  # GKs are filtered out by compute_metrics.py

    # ── Impute missing 360 features ──────────────────────────────────────────
    fill_cols = {
        "n_opponents_5m": 0,
        "n_opponents_10m": 0,
        "nearest_opponent_dist": 15.0,
        "n_teammates_visible": 0,
        "n_opponents_visible": 0,
        "n_open_teammates": 0,
        "n_clear_passing_lanes": 0,
        "score_diff": 0,
    }
    for col, val in fill_cols.items():
        if col in actions.columns:
            actions[col] = actions[col].fillna(val)

    actions["under_pressure"] = actions["under_pressure"].fillna(False).astype(int)
    actions["minute"] = actions["minute"].fillna(0).astype(int)
    actions["period"] = actions["period"].fillna(1).astype(int)

    # ── Select final feature columns ─────────────────────────────────────────
    ID_COLS = ["event_uuid", "match_id", "player", "team", "opponent",
               "event_type", "outcome", "minute", "period"]

    FEATURE_COLS = [
        # Actor / State
        "start_x", "start_y", "xT_origin", "turnover_danger",
        "dist_to_goal", "angle_to_goal_rad",
        "under_pressure", "score_diff",
        # 360 context (from YOUR Pillar 2)
        "n_opponents_5m", "n_opponents_10m", "nearest_opponent_dist",
        "n_teammates_visible", "n_opponents_visible",
        "n_open_teammates", "n_clear_passing_lanes",
        "has_360_data",
        # Action geometry (from BLACKBOX)
        "target_x", "target_y", "target_xT",
        "dx", "dy", "action_distance", "action_angle_rad",
        # One-hot
        "act_pass", "act_carry", "act_shot", "act_dispossessed",
        "role_DEF", "role_MID", "role_ATT", "role_GK",
    ]

    TARGET_COL = "realized_delta_xT"

    # Also keep DQ columns for comparison
    KEEP_COLS = ["delta_xt", "decision_quality", "is_turnover",
                 "goal_within_5_actions", "possession_ends_goal"]

    out_cols = ID_COLS + [TARGET_COL] + FEATURE_COLS + KEEP_COLS
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in actions.columns]

    training_df = actions[out_cols].copy()

    # ── Final cleanup ────────────────────────────────────────────────────────
    # Drop any remaining NaN in features
    feature_cols_present = [c for c in FEATURE_COLS if c in training_df.columns]
    before = len(training_df)
    training_df[feature_cols_present] = training_df[feature_cols_present].fillna(0)
    print(f"  Final dataset: {len(training_df):,} rows × {len(training_df.columns)} columns")

    # ── Save ─────────────────────────────────────────────────────────────────
    training_df.to_csv(out_dir / "ml_training.csv", index=False)
    print(f"\nSaved → {out_dir / 'ml_training.csv'}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n═══ TRAINING DATA SUMMARY ═══")
    print(f"  Total rows:    {len(training_df):,}")
    print(f"  Matches:       {training_df['match_id'].nunique()}")
    print(f"  Teams:         {training_df['team'].nunique()}")
    print(f"  Players:       {training_df['player'].nunique()}")
    print(f"\n  Target (realized_delta_xT):")
    print(f"    Range: [{training_df[TARGET_COL].min():.4f}, {training_df[TARGET_COL].max():.4f}]")
    print(f"    Mean:  {training_df[TARGET_COL].mean():.6f}")
    print(f"    Std:   {training_df[TARGET_COL].std():.6f}")
    print(f"\n  Action breakdown:")
    for at in ["Pass", "Carry", "Shot", "Dispossessed", "Miscontrol"]:
        sub = training_df[training_df["event_type"] == at]
        if len(sub):
            print(f"    {at}: {len(sub):,} ({len(sub)/len(training_df)*100:.1f}%)"
                  f"  avg label: {sub[TARGET_COL].mean():.6f}")


if __name__ == "__main__":
    main()
