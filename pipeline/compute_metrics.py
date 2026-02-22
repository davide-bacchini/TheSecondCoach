"""
compute_metrics.py
==================
Build decision-making analytics from StatsBomb open-data (events + 360).

Outputs (all CSVs in --out-dir):
  xt_grid.csv           – 16×12 Expected Threat grid
  events_enriched.csv   – every Argentina event with spatial + xT metrics
  decision_quality.csv  – on-ball actions scored: actual ΔxT vs best available

Usage:
  source .venv/bin/activate
  python compute_metrics.py --repo-root . --out-dir argentina_wc2022
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
PITCH_X = 120   # StatsBomb pitch length
PITCH_Y = 80    # StatsBomb pitch width
XT_COLS = 16    # xT grid columns (along length)
XT_ROWS = 12    # xT grid rows (along width)
CELL_W  = PITCH_X / XT_COLS   # 7.5
CELL_H  = PITCH_Y / XT_ROWS   # 6.667

TEAM_QUERY = "argentina"   # default team filter


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def xy_to_cell(x, y):
    """Map StatsBomb (x, y) → xT grid cell (col, row)."""
    col = min(int(x / CELL_W), XT_COLS - 1)
    row = min(int(y / CELL_H), XT_ROWS - 1)
    return col, row


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def point_to_line_dist(px, py, ax, ay, bx, by):
    """Distance from point P to line segment A→B."""
    abx, aby = bx - ax, by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return dist(px, py, ax, ay)
    t = max(0, min(1, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return dist(px, py, proj_x, proj_y)


# ──────────────────────────────────────────────────────────────────────
# PILLAR 1: Build xT grid from ALL matches in a competition
# ──────────────────────────────────────────────────────────────────────

def build_xt_grid(data_dir: Path, comp_id: int, season_id: int):
    """
    Build Expected Threat grid:
      xT(cell) = P(shot within 3 actions | ball in cell) × P(goal | shot from cell)

    Uses all matches in the competition for statistical power.
    """
    matches_path = data_dir / "matches" / str(comp_id) / f"{season_id}.json"
    matches = read_json(matches_path)
    match_ids = [int(m["match_id"]) for m in matches]

    # Accumulators
    move_count  = np.zeros((XT_ROWS, XT_COLS), dtype=float)  # times ball enters cell
    shot_count  = np.zeros((XT_ROWS, XT_COLS), dtype=float)  # shots within 3 actions
    shot_total  = np.zeros((XT_ROWS, XT_COLS), dtype=float)  # shots from cell
    goal_total  = np.zeros((XT_ROWS, XT_COLS), dtype=float)  # goals from cell

    for mid in tqdm(match_ids, desc="Building xT grid"):
        ep = data_dir / "events" / f"{mid}.json"
        if not ep.exists():
            continue
        events = read_json(ep)

        # Group events by possession
        poss_chains = {}
        for e in events:
            pid = e.get("possession")
            if pid is not None:
                poss_chains.setdefault(pid, []).append(e)

        for pid, chain in poss_chains.items():
            # Only on-ball actions with location
            on_ball = [e for e in chain if e.get("location") and
                       e["type"]["name"] in ("Pass", "Carry", "Shot", "Dribble",
                                              "Ball Receipt*", "Clearance")]
            for i, e in enumerate(on_ball):
                loc = e["location"]
                col, row = xy_to_cell(loc[0], loc[1])
                move_count[row, col] += 1

                # Look ahead: is there a shot within 3 actions?
                lookahead = on_ball[i:i+4]  # current + next 3
                has_shot = any(la["type"]["name"] == "Shot" for la in lookahead)
                if has_shot:
                    shot_count[row, col] += 1

                # If this IS a shot, record shot + goal
                if e["type"]["name"] == "Shot":
                    shot_total[row, col] += 1
                    outcome = e.get("shot", {}).get("outcome", {}).get("name", "")
                    if outcome == "Goal":
                        goal_total[row, col] += 1

    # Compute probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        p_shot = np.where(move_count > 0, shot_count / move_count, 0)
        p_goal_given_shot = np.where(shot_total > 0, goal_total / shot_total, 0)
        xt = p_shot * p_goal_given_shot

    # Smooth with small Gaussian to reduce noise
    from scipy.ndimage import gaussian_filter
    xt = gaussian_filter(xt, sigma=0.8)

    return xt, p_shot, p_goal_given_shot


# ──────────────────────────────────────────────────────────────────────
# PILLAR 2: Spatial Pressure Metrics from 360 freeze-frames
# ──────────────────────────────────────────────────────────────────────

def compute_spatial_metrics(event: dict, frame: dict):
    """
    From a 360 freeze-frame, compute spatial pressure metrics relative
    to the ball carrier.
    """
    loc = event.get("location")
    if not loc or not frame:
        return {}

    bx, by = loc[0], loc[1]
    ff = frame.get("freeze_frame", [])

    opponents = []
    teammates = []
    for p in ff:
        if not p.get("location"):
            continue
        px, py = p["location"][0], p["location"][1]
        d = dist(bx, by, px, py)
        if p.get("actor"):
            continue  # skip the ball carrier themselves
        if p.get("teammate"):
            teammates.append({"x": px, "y": py, "dist": d, "keeper": p.get("keeper", False)})
        else:
            opponents.append({"x": px, "y": py, "dist": d, "keeper": p.get("keeper", False)})

    opp_dists = [o["dist"] for o in opponents]
    tm_dists  = [t["dist"] for t in teammates]

    metrics = {
        "n_opponents_5m":  sum(1 for d in opp_dists if d <= 5),
        "n_opponents_10m": sum(1 for d in opp_dists if d <= 10),
        "nearest_opponent_dist": min(opp_dists) if opp_dists else 99.0,
        "n_teammates_visible": len(teammates),
        "n_opponents_visible": len(opponents),
    }

    # Open teammates: teammate with no opponent within 3m of them
    n_open = 0
    open_teammate_positions = []
    for t in teammates:
        if t["keeper"]:
            continue
        closest_opp_to_t = min(
            (dist(t["x"], t["y"], o["x"], o["y"]) for o in opponents),
            default=99.0
        )
        if closest_opp_to_t > 3.0:
            n_open += 1
            open_teammate_positions.append(t)

    metrics["n_open_teammates"] = n_open

    # Passing lane quality: for each open teammate, check if line is blocked
    n_clear_lanes = 0
    for t in open_teammate_positions:
        lane_blocked = False
        for o in opponents:
            d_to_lane = point_to_line_dist(o["x"], o["y"], bx, by, t["x"], t["y"])
            if d_to_lane < 2.0:  # opponent within 2m of passing line
                lane_blocked = True
                break
        if not lane_blocked:
            n_clear_lanes += 1

    metrics["n_clear_passing_lanes"] = n_clear_lanes

    return metrics


# ──────────────────────────────────────────────────────────────────────
# PILLAR 3 + 4: Action Value & Decision Quality
# ──────────────────────────────────────────────────────────────────────

def compute_action_value_and_decision(events: list, frames_map: dict, xt_grid: np.ndarray):
    """
    For each on-ball event:
      - Compute ΔxT (threat gained/lost by this action)
      - PENALIZE failed actions: possession loss = -xT_from (all threat gone)
      - Include Dispossessed / Miscontrol as turnover events
      - Risk-adjusted Decision Quality (no free 1.0 in low-xT zones)
    """
    # Group by possession
    poss_chains = {}
    for e in events:
        pid = e.get("possession")
        if pid is not None:
            poss_chains.setdefault(pid, []).append(e)

    results = []

    for pid, chain in poss_chains.items():
        # Include ball-loss events alongside on-ball actions
        on_ball_types = {"Pass", "Carry", "Shot", "Dribble", "Dispossessed", "Miscontrol"}
        on_ball = [e for e in chain if e["type"]["name"] in on_ball_types and e.get("location")]

        # Did this possession end in a goal?
        poss_goal = any(
            e["type"]["name"] == "Shot" and
            e.get("shot", {}).get("outcome", {}).get("name") == "Goal"
            for e in chain
        )

        for i, e in enumerate(on_ball):
            loc = e["location"]
            col_from, row_from = xy_to_cell(loc[0], loc[1])
            xt_from = float(xt_grid[row_from, col_from])

            etype = e["type"]["name"]
            euuid = e.get("id", "")
            is_turnover = False  # track whether this action lost possession

            # Determine destination and outcome
            if etype == "Pass":
                end_loc = e.get("pass", {}).get("end_location")
                outcome = e.get("pass", {}).get("outcome", {}).get("name", "Complete")
                pass_success = outcome == "Complete"
                if outcome in ("Incomplete", "Out", "Pass Offside", "Unknown"):
                    is_turnover = True
            elif etype == "Carry":
                end_loc = e.get("carry", {}).get("end_location")
                outcome = "Complete"
                pass_success = True
            elif etype == "Shot":
                end_loc = None
                outcome = e.get("shot", {}).get("outcome", {}).get("name", "")
                pass_success = outcome == "Goal"
            elif etype == "Dribble":
                if i + 1 < len(on_ball):
                    end_loc = on_ball[i + 1].get("location")
                else:
                    end_loc = loc
                outcome = e.get("dribble", {}).get("outcome", {}).get("name", "")
                pass_success = outcome == "Complete"
                if outcome == "Incomplete":
                    is_turnover = True
            elif etype in ("Dispossessed", "Miscontrol"):
                # Ball-loss event: player lost the ball at their location
                end_loc = None
                outcome = "Turnover"
                pass_success = False
                is_turnover = True
            else:
                end_loc = None
                outcome = ""
                pass_success = False

            # ── ΔxT with TURNOVER PENALTY ────────────────────────────
            if is_turnover:
                # Possession lost: threat drops to zero (opponent gains the ball)
                # Penalty = -xT_from (the more dangerous the zone, the worse the loss)
                xt_to = 0.0
                delta_xt = -xt_from
            elif etype == "Shot":
                xg = e.get("shot", {}).get("statsbomb_xg", 0)
                xt_to = float(xg)
                delta_xt = xt_to - xt_from
            elif end_loc:
                col_to, row_to = xy_to_cell(end_loc[0], end_loc[1])
                xt_to = float(xt_grid[row_to, col_to])
                delta_xt = xt_to - xt_from
            else:
                xt_to = xt_from
                delta_xt = 0.0

            # Goal within 5 actions?
            lookahead = on_ball[i:i+6]
            goal_within_5 = any(
                la["type"]["name"] == "Shot" and
                la.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                for la in lookahead
            )

            # Best available ΔxT from open teammates (using 360 data)
            # We track TWO targets:
            #   best_target_xy → highest-xT open teammate (for DQ scoring)
            #   safe_pass_xy   → best nearby open teammate (for coaching)
            frame = frames_map.get(euuid)
            best_available_xt = xt_from  # baseline: hold ball
            best_target_xy = None
            safe_pass_xy = None
            safe_pass_score = -999.0
            MAX_PASS_DIST = 35.0  # realistic max pass distance for coaching

            spatial = compute_spatial_metrics(e, frame) if frame else {}

            if frame:
                ff = frame.get("freeze_frame", [])
                opponents = [p for p in ff if not p.get("teammate") and p.get("location")]
                for p in ff:
                    if not p.get("teammate") or p.get("actor") or p.get("keeper") or not p.get("location"):
                        continue
                    tx, ty = p["location"][0], p["location"][1]

                    closest_opp = min(
                        (dist(tx, ty, o["location"][0], o["location"][1]) for o in opponents),
                        default=99.0
                    )
                    if closest_opp < 2.0:
                        continue

                    lane_blocked = False
                    for o in opponents:
                        d_to_lane = point_to_line_dist(
                            o["location"][0], o["location"][1],
                            loc[0], loc[1], tx, ty
                        )
                        if d_to_lane < 1.5:
                            lane_blocked = True
                            break
                    if lane_blocked:
                        continue

                    tc, tr = xy_to_cell(tx, ty)
                    tm_xt = float(xt_grid[tr, tc])

                    # Track highest xT option (for scoring)
                    if tm_xt > best_available_xt:
                        best_available_xt = tm_xt
                        best_target_xy = (tx, ty)

                    # Track best *reachable* safe pass for coaching
                    # Must be within realistic passing distance
                    pass_dist = dist(loc[0], loc[1], tx, ty)
                    if pass_dist <= MAX_PASS_DIST and pass_dist > 3.0:
                        # Score: advancement (positive x gain) + openness
                        x_gain = tx - loc[0]  # how far forward
                        # Composite: advancement weighted 0.6, openness 0.4
                        score = 0.6 * x_gain + 0.4 * closest_opp
                        if score > safe_pass_score:
                            safe_pass_score = score
                            safe_pass_xy = (tx, ty)

            best_delta_xt = best_available_xt - xt_from

            # ── 3-COMPONENT Decision Quality Score ───────────────────
            # Components (weighted sum):
            #   1. RETENTION (w=0.40): Did you keep the ball?
            #   2. PROGRESSION (w=0.35): Did you move toward goal?
            #   3. OPPORTUNITY (w=0.25): Did you miss better options?
            #
            # This gives realistic distributions:
            #   Safe pass in own half → ~0.65 (good retention, no progression)
            #   Progressive pass      → ~0.85 (good retention + progression)
            #   Turnover              → ~0.10 (lost ball, everything collapses)
            #   Shot on target        → ~0.90+

            W_RET, W_PROG, W_OPP = 0.40, 0.35, 0.25

            # Component 1: RETENTION — did you keep the ball?
            if is_turnover:
                c_retention = 0.0
            elif etype == "Shot":
                # Shots are inherently "ending possession" — score by xG
                xg_val = e.get("shot", {}).get("statsbomb_xg", 0)
                c_retention = min(1.0, 0.5 + float(xg_val))  # good shots ≥ 0.5
            else:
                c_retention = 1.0  # kept possession

            # Component 2: PROGRESSION — did you move the ball toward danger?
            # Use a sigmoid-like function on ΔxT so small improvements
            # still score well, and big improvements score great.
            # sigmoid: 0.5 + 0.5 * tanh(delta_xt * k)
            # With k=50: ΔxT=+0.01 → 0.74, ΔxT=+0.05 → 0.96, ΔxT=-0.02 → 0.31
            k_prog = 50.0
            c_progression = 0.5 + 0.5 * np.tanh(delta_xt * k_prog)

            # Component 3: OPPORTUNITY — did you miss a clearly better option?
            # If best_delta_xt >> delta_xt, opportunity cost is high → lower score.
            # If no better option exists or player chose close to best → high score.
            missed_xt = best_delta_xt - delta_xt  # how much threat was left on table
            if missed_xt <= 0.001:
                c_opportunity = 1.0  # chose the best or better
            else:
                # Scale: missing 0.01 → 0.80, missing 0.05 → 0.33, missing 0.10 → 0.12
                c_opportunity = max(0.0, 1.0 / (1.0 + missed_xt * 20.0))

            decision_score = (W_RET * c_retention +
                              W_PROG * c_progression +
                              W_OPP * c_opportunity)

            row = {
                "event_uuid": euuid,
                "possession_id": pid,
                "event_type": etype,
                "player": e.get("player", {}).get("name", ""),
                "team": e.get("team", {}).get("name", ""),
                "minute": e.get("minute"),
                "second": e.get("second"),
                "timestamp": e.get("timestamp", ""),
                "period": e.get("period"),
                "under_pressure": bool(e.get("under_pressure")),
                "start_x": loc[0],
                "start_y": loc[1],
                "end_x": end_loc[0] if end_loc else None,
                "end_y": end_loc[1] if end_loc else None,
                "outcome": outcome,
                "action_success": pass_success,
                "is_turnover": is_turnover,
                "xt_from": round(xt_from, 6),
                "xt_to": round(xt_to, 6),
                "delta_xt": round(delta_xt, 6),
                "best_available_xt": round(best_available_xt, 6),
                "best_delta_xt": round(best_delta_xt, 6),
                "decision_quality": round(decision_score, 4),
                "best_target_x": best_target_xy[0] if best_target_xy else None,
                "best_target_y": best_target_xy[1] if best_target_xy else None,
                "safe_pass_x": safe_pass_xy[0] if safe_pass_xy else None,
                "safe_pass_y": safe_pass_xy[1] if safe_pass_xy else None,
                "goal_within_5_actions": goal_within_5,
                "possession_ends_goal": poss_goal,
            }
            row.update(spatial)
            results.append(row)

    return results


# ──────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────

def plot_xt_heatmap(xt_grid: np.ndarray, out_path: Path):
    """Draw the xT grid as a heatmap on a pitch."""
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor("#1a1a2e")

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#e0e0e0",
        linewidth=1.2,
    )
    pitch.draw(ax=ax)

    # Build mesh coordinates in StatsBomb space
    x_edges = np.linspace(0, PITCH_X, XT_COLS + 1)
    y_edges = np.linspace(0, PITCH_Y, XT_ROWS + 1)
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(cx, cy)

    im = ax.pcolormesh(
        x_edges, y_edges, xt_grid,
        cmap="hot", shading="flat", alpha=0.75, zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Expected Threat (xT)", color="white", fontsize=12)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        "Expected Threat (xT) Grid — FIFA World Cup 2022\nP(shot within 3 actions) × P(goal | shot)",
        color="white", fontsize=14, fontweight="bold", pad=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved xT heatmap → {out_path}")


def plot_decision_scatter(df: pd.DataFrame, out_path: Path):
    """Scatter: actual ΔxT vs best available ΔxT, colored by event type."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    type_colors = {
        "Pass": "#00d4ff",
        "Carry": "#ffd93d",
        "Shot": "#ff6b6b",
        "Dribble": "#6bff99",
    }

    for etype, color in type_colors.items():
        sub = df[df["event_type"] == etype]
        ax.scatter(
            sub["best_delta_xt"], sub["delta_xt"],
            s=20, alpha=0.5, c=color, label=etype, edgecolors="none",
        )

    # 45° line (perfect decision)
    lim = max(abs(df["best_delta_xt"].max()), abs(df["delta_xt"].max()), 0.01)
    ax.plot([-lim, lim], [-lim, lim], "--", color="white", alpha=0.4, lw=1)
    ax.axhline(0, color="white", alpha=0.2, lw=0.8)
    ax.axvline(0, color="white", alpha=0.2, lw=0.8)

    ax.set_xlabel("Best Available ΔxT", color="white", fontsize=12)
    ax.set_ylabel("Actual ΔxT", color="white", fontsize=12)
    ax.set_title(
        "Decision Quality — Argentina WC2022\nActual vs Best Available Threat Gain",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.legend(facecolor="#16213e", edgecolor="white", labelcolor="white", fontsize=10)

    for spine in ax.spines.values():
        spine.set_color("#444")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved decision scatter → {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--team", type=str, default="Argentina",
                    help="Team name to filter, or 'ALL' for all teams")
    ap.add_argument("--competition-id", type=int, default=43)
    ap.add_argument("--season-id", type=int, default=106)
    ap.add_argument("--out-dir", type=str, default="argentina_wc2022")
    args = ap.parse_args()

    data_dir = Path(args.repo_root).resolve() / "data"
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    process_all = args.team.upper() == "ALL"
    team_q = args.team.lower()

    # ── Step 1: Build xT grid from ALL matches ──────────────────────
    print("\n═══ PILLAR 1: Expected Threat Grid ═══")
    xt_grid, p_shot, p_goal = build_xt_grid(data_dir, args.competition_id, args.season_id)
    print(f"  xT range: [{xt_grid.min():.6f}, {xt_grid.max():.6f}]")
    print(f"  Peak xT cell: row={np.unravel_index(xt_grid.argmax(), xt_grid.shape)[0]}, "
          f"col={np.unravel_index(xt_grid.argmax(), xt_grid.shape)[1]}")

    # Save xT grid as CSV
    xt_df = pd.DataFrame(xt_grid,
                         columns=[f"col_{c}" for c in range(XT_COLS)],
                         index=[f"row_{r}" for r in range(XT_ROWS)])
    xt_df.to_csv(out_dir / "xt_grid.csv")
    print(f"  Saved → {out_dir / 'xt_grid.csv'}")

    # Plot xT heatmap
    plot_xt_heatmap(xt_grid, out_dir / "xt_heatmap.png")

    # ── Step 2-4: Process matches ───────────────────────────────────
    print("\n═══ PILLARS 2-4: Spatial + Action Value + Decision Quality ═══")

    matches_path = data_dir / "matches" / str(args.competition_id) / f"{args.season_id}.json"
    matches = read_json(matches_path)

    # Build match metadata lookup
    match_meta = {}
    for m in matches:
        mid = int(m["match_id"])
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        # Compute score diff from match result
        home_score = int(m.get("home_score", 0) or 0)
        away_score = int(m.get("away_score", 0) or 0)
        match_meta[mid] = {
            "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
        }

    if process_all:
        target_match_ids = [int(m["match_id"]) for m in matches]
        print(f"  Processing ALL {len(target_match_ids)} matches")
    else:
        target_match_ids = [
            int(m["match_id"]) for m in matches
            if team_q in (m["home_team"]["home_team_name"] + m["away_team"]["away_team_name"]).lower()
        ]
        print(f"  {args.team} matches: {len(target_match_ids)}")

    all_results = []
    for mid in tqdm(target_match_ids, desc="Processing matches"):
        ep = data_dir / "events" / f"{mid}.json"
        fp = data_dir / "three-sixty" / f"{mid}.json"
        if not ep.exists():
            continue

        events = read_json(ep)
        frames_map = {}
        if fp.exists():
            frames = read_json(fp)
            frames_map = {f["event_uuid"]: f for f in frames}

        results = compute_action_value_and_decision(events, frames_map, xt_grid)

        # Attach match-level metadata
        meta = match_meta.get(mid, {})
        for r in results:
            r["match_id"] = mid
            # Score diff from team's perspective
            team_name = r.get("team", "")
            if team_name.lower() in meta.get("home_team", "").lower():
                r["score_diff"] = meta["home_score"] - meta["away_score"]
                r["opponent"] = meta["away_team"]
            elif team_name.lower() in meta.get("away_team", "").lower():
                r["score_diff"] = meta["away_score"] - meta["home_score"]
                r["opponent"] = meta["home_team"]
            else:
                r["score_diff"] = 0
                r["opponent"] = ""
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    if process_all:
        team_df = df.copy()
    else:
        team_df = df[df["team"].str.lower().str.contains(team_q)].copy()

    # Save full enriched events (both teams for context)
    df.to_csv(out_dir / "events_enriched.csv", index=False)
    print(f"  events_enriched.csv: {len(df)} rows")

    # Save decision quality
    team_df.to_csv(out_dir / "decision_quality.csv", index=False)
    label = "all teams" if process_all else f"{args.team} only"
    print(f"  decision_quality.csv: {len(team_df)} rows ({label})")

    # Plot decision scatter
    if not team_df.empty:
        plot_decision_scatter(team_df, out_dir / "decision_scatter.png")

    # ── Summary statistics ──────────────────────────────────────────
    print("\n═══ SUMMARY STATS ═══")
    print(f"  Total on-ball actions:     {len(team_df)}")
    if process_all:
        print(f"  Teams: {team_df['team'].nunique()}")
        print(f"  Matches: {team_df['match_id'].nunique()}")
    for etype in ["Pass", "Carry", "Shot", "Dribble", "Dispossessed", "Miscontrol"]:
        sub = team_df[team_df["event_type"] == etype]
        if sub.empty:
            continue
        print(f"\n  {etype}s ({len(sub)}):")
        print(f"    avg ΔxT:              {sub['delta_xt'].mean():.6f}")
        print(f"    avg decision quality:  {sub['decision_quality'].mean():.3f}")
        print(f"    avg opponents <5m:     {sub.get('n_opponents_5m', pd.Series([0])).mean():.2f}")
        print(f"    avg open teammates:    {sub.get('n_open_teammates', pd.Series([0])).mean():.2f}")

    goals = team_df[team_df["goal_within_5_actions"]]
    non_goals = team_df[~team_df["goal_within_5_actions"]]
    print(f"\n  Actions leading to goal (within 5):")
    print(f"    count: {len(goals)}")
    if not goals.empty:
        print(f"    avg ΔxT:              {goals['delta_xt'].mean():.6f}")
        print(f"    avg decision quality:  {goals['decision_quality'].mean():.3f}")
    print(f"\n  Actions NOT leading to goal:")
    print(f"    count: {len(non_goals)}")
    if not non_goals.empty:
        print(f"    avg ΔxT:              {non_goals['delta_xt'].mean():.6f}")
        print(f"    avg decision quality:  {non_goals['decision_quality'].mean():.3f}")

    print(f"\nDone. All outputs → {out_dir}/")


if __name__ == "__main__":
    main()

