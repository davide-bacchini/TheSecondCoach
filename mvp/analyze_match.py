"""
analyze_match.py  (Phase 4 — Inference Pipeline)
=================================================
Given any match (from any competition with 360 data):
 1. Process events + 360 through compute_metrics pipeline
 2. Generate candidate actions per event from freeze-frame data
 3. Score all candidates with the trained XGBoost model
 4. Pick the ML-predicted best action → coaching recommendation
 5. Generate per-player visual reports (same style as Argentina WC2022)

Usage:
  source .venv/bin/activate
  python analyze_match.py --match-id 3930176 --comp-id 55 --season-id 282 \\
      --model wc2022/models/vaep_xgb.json --xt-grid wc2022/xt_grid.csv
"""

import argparse
import json
import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# ── Constants ────────────────────────────────────────────────────────────────
PITCH_X, PITCH_Y = 120.0, 80.0
XT_COLS, XT_ROWS = 16, 12
GOAL_X, GOAL_Y = 120.0, 40.0

BG       = '#0f0f23'
PITCH_BG = '#1a1a2e'
GREEN    = '#2ecc71'
RED      = '#e74c3c'
GOLD     = '#f1c40f'
WHITE    = '#ecf0f1'
ORANGE   = '#e67e22'
CYAN     = '#00d4ff'
BLUE_TM  = '#3498db'
RED_OPP  = '#e74c3c'
LIME     = '#00ff88'
MAGENTA  = '#e040fb'

# Model feature list (must match train_model.py)
FEATURES = [
    "start_x", "start_y", "xT_origin", "turnover_danger",
    "dist_to_goal", "angle_to_goal_rad", "under_pressure", "score_diff",
    "n_opponents_5m", "n_opponents_10m", "nearest_opponent_dist",
    "n_teammates_visible", "n_opponents_visible",
    "n_open_teammates", "n_clear_passing_lanes", "has_360_data",
    "target_x", "target_y", "target_xT",
    "dx", "dy", "action_distance", "action_angle_rad",
    "act_pass", "act_carry", "act_shot", "act_dispossessed",
    "role_DEF", "role_MID", "role_ATT", "role_GK",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pdist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def xT_lookup(grid, x, y):
    xi = int(np.clip(x / PITCH_X * XT_COLS, 0, XT_COLS - 1))
    yi = int(np.clip(y / PITCH_Y * XT_ROWS, 0, XT_ROWS - 1))
    return float(grid[yi, xi])


def angle_to_goal(x, y):
    a = np.array([GOAL_X - x, 36.0 - y])
    b = np.array([GOAL_X - x, 44.0 - y])
    cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.arccos(np.clip(cos_t, -1, 1)))


def point_to_line_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return pdist(px, py, ax, ay)
    t = max(0, min(1, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    return pdist(px, py, ax + t * abx, ay + t * aby)


def short_name(full_name):
    parts = full_name.split()
    return parts[-1] if len(parts) >= 2 else full_name


def make_safe_name(full_name):
    parts = full_name.split()
    name = f"{parts[0]}_{parts[-1]}".lower() if len(parts) >= 2 else full_name.lower()
    return re.sub(r"[^a-z0-9_]", "", name)


def needs_mirror(event_type, frame, csv_sx, csv_sy):
    if not frame:
        return False
    ff = frame.get("freeze_frame", [])
    actor = next((p for p in ff if p.get("actor") and p.get("location")), None)
    if not actor:
        return False
    ax, ay = actor["location"]
    orig_d = pdist(csv_sx, csv_sy, ax, ay)
    mirror_d = pdist(csv_sx, csv_sy, 120 - ax, 80 - ay)
    return mirror_d < orig_d - 5


def mirror_frame(frame):
    new_ff = []
    for p in frame.get("freeze_frame", []):
        np_ = dict(p)
        if p.get("location"):
            np_["location"] = [120 - p["location"][0], 80 - p["location"][1]]
        if not p.get("actor"):
            np_["teammate"] = not p.get("teammate", False)
        new_ff.append(np_)
    return {"freeze_frame": new_ff}


# ═══════════════════════════════════════════════════════════════════════════════
# ML-powered candidate generation and scoring
# ═══════════════════════════════════════════════════════════════════════════════

def generate_candidates(sx, sy, frame, spatial, score_diff, under_pressure, xt_grid):
    """
    Generate candidate actions for a given position and freeze frame.
    Returns list of dicts, each with features ready for the model.
    """
    if not frame:
        return []

    ff = frame.get("freeze_frame", [])
    opponents = [p for p in ff if not p.get("teammate") and p.get("location")]
    teammates = [p for p in ff if p.get("teammate") and not p.get("actor") and p.get("location")]

    xt_origin = xT_lookup(xt_grid, sx, sy)
    td = xT_lookup(xt_grid, PITCH_X - sx, sy)
    dg = pdist(sx, sy, GOAL_X, GOAL_Y)
    ag = angle_to_goal(sx, sy)

    # Spatial metrics from frame
    n_opp_5 = spatial.get("n_opponents_5m", 0)
    n_opp_10 = spatial.get("n_opponents_10m", 0)
    near_opp = spatial.get("nearest_opponent_dist", 15.0)
    n_tm = spatial.get("n_teammates_visible", 0)
    n_opp_vis = spatial.get("n_opponents_visible", 0)
    n_open = spatial.get("n_open_teammates", 0)
    n_clear = spatial.get("n_clear_passing_lanes", 0)

    base = {
        "start_x": sx, "start_y": sy,
        "xT_origin": xt_origin, "turnover_danger": td,
        "dist_to_goal": dg, "angle_to_goal_rad": ag,
        "under_pressure": int(under_pressure),
        "score_diff": score_diff,
        "n_opponents_5m": n_opp_5,
        "n_opponents_10m": n_opp_10,
        "nearest_opponent_dist": near_opp,
        "n_teammates_visible": n_tm,
        "n_opponents_visible": n_opp_vis,
        "n_open_teammates": n_open,
        "n_clear_passing_lanes": n_clear,
        "has_360_data": 1,
        "role_DEF": int(sx < 40),
        "role_MID": int(40 <= sx < 80),
        "role_ATT": int(sx >= 80),
        "role_GK": 0,
    }

    candidates = []

    # ── Pass candidates: to each visible teammate ────────────────────────────
    for t in teammates:
        tx, ty = clamp(t["location"][0], 0.5, 119.5), clamp(t["location"][1], 0.5, 79.5)
        d = pdist(sx, sy, tx, ty)
        if d < 3.0 or d > 80.0:
            continue

        # Check lane blockage
        lane_blocked = False
        for o in opponents:
            ox, oy = o["location"][0], o["location"][1]
            if point_to_line_dist(ox, oy, sx, sy, tx, ty) < 2.0:
                lane_blocked = True
                break

        dx, dy = tx - sx, ty - sy
        cand = dict(base)
        cand.update({
            "target_x": tx, "target_y": ty,
            "target_xT": xT_lookup(xt_grid, tx, ty),
            "dx": dx, "dy": dy,
            "action_distance": math.sqrt(dx ** 2 + dy ** 2),
            "action_angle_rad": math.atan2(dy, dx),
            "act_pass": 1, "act_carry": 0, "act_shot": 0, "act_dispossessed": 0,
            "action_label": "pass",
            "is_keeper_pass": bool(t.get("keeper")),
            "lane_blocked": lane_blocked,
            "is_long": d > 30,
        })
        candidates.append(cand)

    # ── Carry candidates: 8 directions, 8m each ─────────────────────────────
    carry_dist = 8.0
    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg)
        tx = sx + carry_dist * math.cos(rad)
        ty = sy + carry_dist * math.sin(rad)
        if tx < 2 or tx > 118 or ty < 2 or ty > 78:
            continue

        # Check path density
        mid_x = sx + (carry_dist * 0.5) * math.cos(rad)
        mid_y = sy + (carry_dist * 0.5) * math.sin(rad)
        path_opps = sum(1 for o in opponents
                        if pdist(o["location"][0], o["location"][1], mid_x, mid_y) < 4)
        dest_opps = sum(1 for o in opponents
                        if pdist(o["location"][0], o["location"][1], tx, ty) < 6)

        dx, dy = tx - sx, ty - sy
        cand = dict(base)
        cand.update({
            "target_x": tx, "target_y": ty,
            "target_xT": xT_lookup(xt_grid, tx, ty),
            "dx": dx, "dy": dy,
            "action_distance": carry_dist,
            "action_angle_rad": rad,
            "act_pass": 0, "act_carry": 1, "act_shot": 0, "act_dispossessed": 0,
            "action_label": "carry",
            "is_keeper_pass": False,
            "lane_blocked": False,
            "is_long": False,
            "path_blocked": path_opps > 1 or dest_opps > 2,
        })
        candidates.append(cand)

    # ── Shot candidate (if close enough) ─────────────────────────────────────
    if dg < 35:
        dx, dy = GOAL_X - sx, GOAL_Y - sy
        cand = dict(base)
        cand.update({
            "target_x": GOAL_X, "target_y": GOAL_Y,
            "target_xT": xT_lookup(xt_grid, GOAL_X, GOAL_Y),
            "dx": dx, "dy": dy,
            "action_distance": dg,
            "action_angle_rad": math.atan2(dy, dx),
            "act_pass": 0, "act_carry": 0, "act_shot": 1, "act_dispossessed": 0,
            "action_label": "shot",
            "is_keeper_pass": False,
            "lane_blocked": False,
            "is_long": False,
        })
        candidates.append(cand)

    return candidates


def score_candidates(candidates, model):
    """Score all candidates with the trained model, return best."""
    if not candidates:
        return None

    cdf = pd.DataFrame(candidates)
    X = cdf[FEATURES].fillna(0)
    cdf["predicted_delta_xT"] = model.predict(X)

    # Penalise blocked lanes for passes
    for i, c in enumerate(candidates):
        if c.get("lane_blocked"):
            cdf.loc[i, "predicted_delta_xT"] -= 0.005
        if c.get("path_blocked"):
            cdf.loc[i, "predicted_delta_xT"] -= 0.003

    best_idx = cdf["predicted_delta_xT"].idxmax()
    best = candidates[best_idx]
    best["predicted_delta_xT"] = cdf.loc[best_idx, "predicted_delta_xT"]
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Event processing (like compute_metrics.py but for a single match)
# ═══════════════════════════════════════════════════════════════════════════════

def process_match_events(events, frames_map, xt_grid, match_meta):
    """Process all events for a single match → DataFrame with DQ + ML recommendation."""
    on_ball_types = {"Pass", "Carry", "Shot", "Dribble", "Dispossessed", "Miscontrol"}

    poss_chains = {}
    for e in events:
        pid = e.get("possession")
        if pid is not None:
            poss_chains.setdefault(pid, []).append(e)

    rows = []
    for pid, chain in poss_chains.items():
        on_ball = [e for e in chain if e["type"]["name"] in on_ball_types and e.get("location")]
        poss_goal = any(
            e["type"]["name"] == "Shot"
            and e.get("shot", {}).get("outcome", {}).get("name") == "Goal"
            for e in chain
        )

        for i, e in enumerate(on_ball):
            loc = e["location"]
            sx, sy = loc[0], loc[1]
            etype = e["type"]["name"]
            euuid = e.get("id", "")
            is_turnover = False

            if etype == "Pass":
                end_loc = e.get("pass", {}).get("end_location")
                outcome = e.get("pass", {}).get("outcome", {}).get("name", "Complete")
                if outcome in ("Incomplete", "Out", "Pass Offside", "Unknown"):
                    is_turnover = True
            elif etype == "Carry":
                end_loc = e.get("carry", {}).get("end_location")
                outcome = "Complete"
            elif etype == "Shot":
                end_loc = None
                outcome = e.get("shot", {}).get("outcome", {}).get("name", "")
            elif etype == "Dribble":
                end_loc = on_ball[i + 1].get("location") if i + 1 < len(on_ball) else loc
                outcome = e.get("dribble", {}).get("outcome", {}).get("name", "")
                if outcome == "Incomplete":
                    is_turnover = True
            elif etype in ("Dispossessed", "Miscontrol"):
                end_loc = None
                outcome = "Turnover"
                is_turnover = True
            else:
                continue

            # Compute spatial metrics from 360
            frame = frames_map.get(euuid)
            n_opp_5m = n_opp_10m = 0
            nearest_opp = 99.0
            n_tm = n_opp = n_open = n_clear = 0
            has_360 = 0

            if frame:
                # Fix coordinate mirroring
                if needs_mirror(etype, frame, sx, sy):
                    frame = mirror_frame(frame)
                    frames_map[euuid] = frame  # cache corrected

                has_360 = 1
                ff = frame.get("freeze_frame", [])
                opponents, teammates = [], []
                for p in ff:
                    if not p.get("location") or p.get("actor"):
                        continue
                    px, py = p["location"]
                    d = pdist(sx, sy, px, py)
                    if p.get("teammate"):
                        teammates.append({"x": px, "y": py, "dist": d, "keeper": p.get("keeper", False)})
                    else:
                        opponents.append({"x": px, "y": py, "dist": d})

                opp_dists = [o["dist"] for o in opponents]
                n_opp_5m = sum(1 for d in opp_dists if d <= 5)
                n_opp_10m = sum(1 for d in opp_dists if d <= 10)
                nearest_opp = min(opp_dists) if opp_dists else 99.0
                n_tm = len(teammates)
                n_opp = len(opponents)

                for t in teammates:
                    if t.get("keeper"):
                        continue
                    closest_o = min(
                        (pdist(t["x"], t["y"], o["x"], o["y"]) for o in opponents),
                        default=99.0,
                    )
                    if closest_o > 3.0:
                        n_open += 1
                        blocked = any(
                            point_to_line_dist(o["x"], o["y"], sx, sy, t["x"], t["y"]) < 2.0
                            for o in opponents
                        )
                        if not blocked:
                            n_clear += 1

            # ΔxT from actual action
            xt_from = xT_lookup(xt_grid, sx, sy)
            if end_loc:
                ex, ey = end_loc[0], end_loc[1]
                xt_to = xT_lookup(xt_grid, ex, ey)
                delta_xt = xt_to - xt_from
            elif etype == "Shot" and "Goal" in str(outcome):
                ex, ey = GOAL_X, GOAL_Y
                delta_xt = 1.0 - xt_from
            else:
                ex, ey = None, None
                delta_xt = -xt_from if is_turnover else 0.0

            if is_turnover and etype == "Pass":
                delta_xt = -xt_from * 0.5

            # Goal within 5 actions
            goal_in_5 = any(
                la["type"]["name"] == "Shot"
                and la.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                for la in on_ball[i : i + 6]
            )

            player = e.get("player", {}).get("name", "")
            team = e.get("team", {}).get("name", "")

            # Score diff
            if team.lower() in match_meta.get("home_team", "").lower():
                sd = match_meta["home_score"] - match_meta["away_score"]
            elif team.lower() in match_meta.get("away_team", "").lower():
                sd = match_meta["away_score"] - match_meta["home_score"]
            else:
                sd = 0

            spatial = {
                "n_opponents_5m": n_opp_5m,
                "n_opponents_10m": n_opp_10m,
                "nearest_opponent_dist": nearest_opp,
                "n_teammates_visible": n_tm,
                "n_opponents_visible": n_opp,
                "n_open_teammates": n_open,
                "n_clear_passing_lanes": n_clear,
            }

            # 3-component decision quality (same as compute_metrics.py)
            W_RET, W_PROG, W_OPP = 0.40, 0.35, 0.25
            if is_turnover:
                c_ret = 0.0
            elif etype == "Shot":
                xg = e.get("shot", {}).get("statsbomb_xg", 0)
                c_ret = min(1.0, 0.5 + float(xg or 0))
            else:
                c_ret = 1.0
            c_prog = 0.5 + 0.5 * np.tanh(delta_xt * 50.0)
            c_opp = 1.0 if delta_xt >= 0 else max(0.0, 1.0 / (1.0 + abs(delta_xt) * 20.0))
            decision_quality = W_RET * c_ret + W_PROG * c_prog + W_OPP * c_opp

            rows.append({
                "event_uuid": euuid,
                "player": player,
                "team": team,
                "event_type": etype,
                "outcome": outcome,
                "minute": e.get("minute", 0),
                "second": e.get("second", 0),
                "period": e.get("period", 1),
                "start_x": sx, "start_y": sy,
                "end_x": ex, "end_y": ey,
                "xt_from": round(xt_from, 6),
                "delta_xt": round(delta_xt, 6),
                "decision_quality": round(decision_quality, 4),
                "is_turnover": is_turnover,
                "under_pressure": bool(e.get("under_pressure")),
                "goal_within_5_actions": goal_in_5,
                "possession_ends_goal": poss_goal,
                "score_diff": sd,
                "match_id": match_meta["match_id"],
                "has_360_data": has_360,
                **spatial,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization (same style as generate_player_reports.py but ML-powered)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_action_pitch(ax, row, frame_data, player_name, is_best, model, xt_grid, match_label):
    """Draw one action on a pitch with ML-powered recommendation."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG, line_color="#444", linewidth=0.8)
    pitch.draw(ax=ax)

    sx, sy = row["start_x"], row["start_y"]
    short_p = short_name(player_name)
    has_frame = frame_data is not None

    # Fix coordinate mirroring
    if has_frame and needs_mirror(row["event_type"], frame_data, sx, sy):
        frame_data = mirror_frame(frame_data)

    num_players = len(frame_data.get("freeze_frame", [])) if has_frame else 0

    # Plot players from freeze frame
    if has_frame:
        for p in frame_data.get("freeze_frame", []):
            if not p.get("location"):
                continue
            px = clamp(p["location"][0], 0.5, 119.5)
            py = clamp(p["location"][1], 0.5, 79.5)
            if p.get("actor"):
                ax.scatter(px, py, s=350, c=CYAN, edgecolors=WHITE, linewidths=2.5, zorder=8)
                label_y = py - 4 if py < 40 else py + 4
                ax.text(px, label_y, short_p, color=CYAN, fontsize=9, fontweight="bold",
                        ha="center", va="center", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.9,
                                  edgecolor=CYAN, linewidth=0.8))
            elif p.get("teammate"):
                if p.get("keeper"):
                    ax.scatter(px, py, s=120, c="#f39c12", edgecolors="#333", linewidths=1.5, zorder=6, marker="D")
                    ax.text(px, py - 3, "GK", color="#f39c12", fontsize=7, ha="center", va="top", fontweight="bold", zorder=7)
                else:
                    ax.scatter(px, py, s=120, c=BLUE_TM, edgecolors=WHITE, linewidths=1, zorder=6)
            else:
                if p.get("keeper"):
                    ax.scatter(px, py, s=120, c="#ff4444", edgecolors="#333", linewidths=1.5, zorder=6, marker="D")
                    ax.text(px, py - 3, "GK", color="#ff4444", fontsize=7, ha="center", va="top", fontweight="bold", zorder=7)
                else:
                    ax.scatter(px, py, s=100, c=RED_OPP, edgecolors=WHITE, linewidths=1, zorder=6, marker="^")
    else:
        ax.scatter(sx, sy, s=350, c=CYAN, edgecolors=WHITE, linewidths=2.5, zorder=8)
        label_y = sy - 4 if sy < 40 else sy + 4
        ax.text(sx, label_y, short_p, color=CYAN, fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.9,
                          edgecolor=CYAN, linewidth=0.8))

    # ── Draw actual action arrow ─────────────────────────────────────────────
    ex, ey = row.get("end_x"), row.get("end_y")
    has_actual = pd.notna(ex) and pd.notna(ey)
    if has_actual:
        ex_c = clamp(float(ex), 0.5, 119.5)
        ey_c = clamp(float(ey), 0.5, 79.5)
        ac_col = GREEN if is_best else (ORANGE if row.get("is_turnover") else RED)
        ax.annotate("", xy=(ex_c, ey_c), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color=ac_col, lw=3, alpha=0.9), zorder=9)
        ax.scatter(ex_c, ey_c, s=100, c=ac_col, alpha=0.7, zorder=9, edgecolors="none")

    # ── ML-powered recommendation ────────────────────────────────────────────
    spatial = {
        "n_opponents_5m": row.get("n_opponents_5m", 0),
        "n_opponents_10m": row.get("n_opponents_10m", 0),
        "nearest_opponent_dist": row.get("nearest_opponent_dist", 15),
        "n_teammates_visible": row.get("n_teammates_visible", 0),
        "n_opponents_visible": row.get("n_opponents_visible", 0),
        "n_open_teammates": row.get("n_open_teammates", 0),
        "n_clear_passing_lanes": row.get("n_clear_passing_lanes", 0),
    }

    candidates = generate_candidates(
        sx, sy, frame_data, spatial,
        row.get("score_diff", 0),
        row.get("under_pressure", False),
        xt_grid,
    )

    best_cand = score_candidates(candidates, model) if candidates else None

    # Draw recommendation
    coaching = ""
    if best_cand:
        btx, bty = best_cand["target_x"], best_cand["target_y"]
        btype = best_cand["action_label"]
        pred = best_cand["predicted_delta_xT"]

        # Don't recommend same thing they did
        same_as_actual = (has_actual and pdist(btx, bty, float(ex or 0), float(ey or 0)) < 8.0
                          and btype == ("pass" if row["event_type"] == "Pass" else
                                        "carry" if row["event_type"] == "Carry" else "shot"))

        if same_as_actual and is_best:
            coaching = ""  # Great decision, no correction
        elif same_as_actual:
            coaching = f"⭐ Right idea, poor execution\n   Timing or weight was off"
        elif btype == "carry":
            ax.annotate("", xy=(btx, bty), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=LIME, lw=2.5, alpha=0.8,
                                        linestyle="dashed"), zorder=8)
            ax.scatter(btx, bty, s=150, c=LIME, marker="D", edgecolors=WHITE,
                       linewidths=1, zorder=9)
            direction = "forward" if btx > sx + 3 else ("backward" if btx < sx - 3 else "wide")
            ax.text(btx, (bty - 3.5 if bty < 40 else bty + 3.5), "Carry here",
                    color=LIME, fontsize=8, ha="center", va="center", fontweight="bold",
                    zorder=10, bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.85,
                                         edgecolor=LIME, linewidth=0.5))
            coaching = f"⭐ Carry {direction} into space\n   ML predicted ΔxT: {pred:+.4f}"

        elif btype == "shot":
            ax.annotate("", xy=(btx, bty), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5, alpha=0.8,
                                        linestyle="dashed"), zorder=8)
            ax.scatter(btx, bty, s=200, c=GOLD, marker="*", edgecolors=WHITE,
                       linewidths=1, zorder=9)
            ax.text(btx - 5, bty, "Shoot!", color=GOLD, fontsize=8, ha="center",
                    fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.85,
                              edgecolor=GOLD, linewidth=0.5))
            coaching = f"⭐ Should have shot!\n   ML predicted ΔxT: {pred:+.4f}"

        else:  # pass
            ax.annotate("", xy=(btx, bty), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.5, alpha=0.8,
                                        linestyle="dashed"), zorder=8)
            ax.scatter(btx, bty, s=200, c=GOLD, marker="*", edgecolors=WHITE,
                       linewidths=1, zorder=9)
            direction = "forward" if btx > sx + 5 else ("backward" if btx < sx - 5 else "wide")
            is_gk = best_cand.get("is_keeper_pass", False)
            label = "Pass to GK" if is_gk else ("Long ball ⚠️" if best_cand.get("is_long") else "Pass here")
            ax.text(btx, (bty - 3.5 if bty < 40 else bty + 3.5), label,
                    color=GOLD, fontsize=8, ha="center", va="center", fontweight="bold",
                    zorder=10, bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.85,
                                          edgecolor=GOLD, linewidth=0.5))
            n_open = int(row.get("n_open_teammates", 0))
            if is_gk:
                coaching = f"⭐ Pass back to GK — safest option\n   ML predicted ΔxT: {pred:+.4f}"
            else:
                coaching = f"⭐ Should have passed {direction}\n   {n_open} open teammate(s)  |  ML ΔxT: {pred:+.4f}"
    elif not has_frame:
        coaching = "⭐ No 360° data — cannot determine best option"

    # ── Title ────────────────────────────────────────────────────────────────
    title_extra = ""
    if not has_frame:
        title_extra = "  [no 360 data]"
    elif num_players < 18:
        title_extra = "  [some players off-camera]"

    mn = f"{int(row['minute'])}'{int(row.get('second', 0)):02d}\""
    if is_best:
        g = " → GOAL!" if row.get("goal_within_5_actions") else ""
        ax.set_title(
            f"{row['event_type']}  |  {match_label} {mn}  |  ΔxT={row['delta_xt']:+.4f}  |  "
            f"DQ={row['decision_quality']:.2f}{g}{title_extra}",
            color=GREEN, fontsize=9, fontweight="bold", pad=4,
        )
    else:
        to = " 💥TURNOVER" if row.get("is_turnover") else ""
        pr = " ⚡PRESS" if row.get("under_pressure") else ""
        ax.set_title(
            f"{row['event_type']} → {row.get('outcome', '')}  |  {match_label} {mn}  |  "
            f"ΔxT={row['delta_xt']:+.4f}  |  DQ={row['decision_quality']:.2f}{to}{pr}{title_extra}",
            color=RED, fontsize=9, fontweight="bold", pad=4,
        )

    # ── Coaching annotation (worst actions only) ─────────────────────────────
    if not is_best and coaching:
        ax.text(2, 3, coaching, color=GOLD, fontsize=8.5, va="bottom", fontweight="bold",
                zorder=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.9,
                                      edgecolor=GOLD, linewidth=0.5))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="ML-powered match analysis and per-player coaching")
    ap.add_argument("--match-id", type=int, required=True)
    ap.add_argument("--comp-id", type=int, default=55)
    ap.add_argument("--season-id", type=int, default=282)
    ap.add_argument("--model", type=str, default=str(Path(__file__).resolve().parent.parent / "models" / "vaep_xgb.json"))
    ap.add_argument("--xt-grid", type=str, default=str(Path(__file__).resolve().parent.parent / "models" / "xt_grid.csv"))
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--team", type=str, default=None,
                    help="Only generate reports for this team (default: both)")
    ap.add_argument("--min-actions", type=int, default=20,
                    help="Minimum actions for a player report")
    args = ap.parse_args()

    data_dir = Path(args.repo_root) / "data"

    # ── Load model and xT grid ───────────────────────────────────────────────
    print("Loading model and xT grid ...")
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(args.model)
    xt_grid = pd.read_csv(args.xt_grid, index_col=0).values
    print(f"  Model: {args.model}")
    print(f"  xT grid: {xt_grid.shape}")

    # ── Load match data ──────────────────────────────────────────────────────
    print("Loading match data ...")
    matches = read_json(data_dir / "matches" / str(args.comp_id) / f"{args.season_id}.json")
    m_info = next(m for m in matches if int(m["match_id"]) == args.match_id)

    home = m_info["home_team"]["home_team_name"]
    away = m_info["away_team"]["away_team_name"]
    hs = int(m_info.get("home_score", 0) or 0)
    ask = int(m_info.get("away_score", 0) or 0)
    match_meta = {
        "match_id": args.match_id,
        "home_team": home, "away_team": away,
        "home_score": hs, "away_score": ask,
    }

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        safe_home = re.sub(r'[^a-z0-9]', '', home.lower())
        safe_away = re.sub(r'[^a-z0-9]', '', away.lower())
        out_dir = Path(f"match_analysis_{safe_home}_vs_{safe_away}")
    out_dir.mkdir(parents=True, exist_ok=True)
    players_dir = out_dir / "players"
    players_dir.mkdir(exist_ok=True)

    print(f"  {home} {hs} - {ask} {away}")

    events = read_json(data_dir / "events" / f"{args.match_id}.json")
    frames_list = read_json(data_dir / "three-sixty" / f"{args.match_id}.json")
    frames_map = {f["event_uuid"]: f for f in frames_list}
    print(f"  Events: {len(events)}, 360 frames: {len(frames_list)}")

    # ── Process events ───────────────────────────────────────────────────────
    print("\nProcessing events ...")
    df = process_match_events(events, frames_map, xt_grid, match_meta)
    print(f"  {len(df)} on-ball actions")
    print(f"  Teams: {df['team'].unique().tolist()}")

    # Save processed data
    df.to_csv(out_dir / "decision_quality.csv", index=False)

    # ── Team filter ──────────────────────────────────────────────────────────
    if args.team:
        team_q = args.team.lower()
        df = df[df["team"].str.lower().str.contains(team_q)].copy()
        print(f"  Filtered to {args.team}: {len(df)} actions")

    # ── Generate per-player reports ──────────────────────────────────────────
    print("\nGenerating per-player reports ...")
    outfield = df[~df["player"].str.contains("Keeper|keeper|Goalkeeper", case=False, na=False)].copy()
    active = outfield.groupby("player").filter(lambda x: len(x) >= args.min_actions)
    player_names = sorted(active["player"].unique())
    print(f"  {len(player_names)} players with >= {args.min_actions} actions")

    match_label = f"{home} {hs}-{ask} {away}"

    # Player summary table
    summary_rows = []

    for player_name in tqdm(player_names, desc="Player reports"):
        p_df = outfield[outfield["player"] == player_name].copy()
        safe_name = make_safe_name(player_name)
        team_name = p_df["team"].iloc[0]

        # Select best/worst
        has_frame_mask = p_df["event_uuid"].isin(frames_map)
        is_goal_mask = p_df["goal_within_5_actions"].astype(bool)
        valid_mask = has_frame_mask | is_goal_mask
        valid_df = p_df[valid_mask]
        pool = valid_df if len(valid_df) >= 10 else p_df
        best_5 = pool.nlargest(5, "delta_xt")
        worst_5 = pool.nsmallest(5, "decision_quality")

        # Create report
        fig, axes = plt.subplots(5, 2, figsize=(24, 42), facecolor=BG)
        fig.subplots_adjust(hspace=0.18, wspace=0.08)
        fig.suptitle(player_name, color=WHITE, fontsize=30, fontweight="bold", y=0.98)

        avg_dq = p_df["decision_quality"].mean()
        total = len(p_df)
        turnovers = int(p_df["is_turnover"].sum())
        to_pct = p_df["is_turnover"].mean() * 100
        goals = int(p_df["goal_within_5_actions"].sum())

        fig.text(0.5, 0.968,
                 f"{team_name} — {match_label}  |  {total} actions  |  "
                 f"Avg DQ: {avg_dq:.2f}  |  Turnovers: {turnovers} ({to_pct:.0f}%)  |  Goals: {goals}",
                 color="#999", fontsize=13, ha="center")
        fig.text(0.27, 0.955, "✅  TOP 5 BEST ACTIONS", color=GREEN, fontsize=16,
                 fontweight="bold", ha="center")
        fig.text(0.73, 0.955, "❌  TOP 5 WORST DECISIONS (ML-powered)", color=RED, fontsize=16,
                 fontweight="bold", ha="center")

        for rank_i, (_, row) in enumerate(best_5.iterrows()):
            draw_action_pitch(axes[rank_i, 0], row, frames_map.get(row["event_uuid"]),
                              player_name, True, model, xt_grid, match_label)
            axes[rank_i, 0].text(2, 77, f"#{rank_i + 1}", color=GREEN, fontsize=16,
                                 fontweight="bold", va="top",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.9,
                                           edgecolor=GREEN), zorder=10)

        for rank_i, (_, row) in enumerate(worst_5.iterrows()):
            draw_action_pitch(axes[rank_i, 1], row, frames_map.get(row["event_uuid"]),
                              player_name, False, model, xt_grid, match_label)
            axes[rank_i, 1].text(2, 77, f"#{rank_i + 1}", color=RED, fontsize=16,
                                 fontweight="bold", va="top",
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.9,
                                           edgecolor=RED), zorder=10)

        # Legend
        legend_elements = [
            plt.scatter([], [], s=200, c=CYAN, edgecolors=WHITE, linewidths=2.5, marker="o",
                        label=f"Ball carrier ({short_name(player_name)})"),
            plt.scatter([], [], s=100, c=BLUE_TM, edgecolors=WHITE, linewidths=1, marker="o",
                        label="Teammates"),
            plt.scatter([], [], s=100, c=RED_OPP, edgecolors=WHITE, linewidths=1, marker="^",
                        label="Opponents"),
            plt.scatter([], [], s=100, c="#f39c12", edgecolors="#333", linewidths=1.5, marker="D",
                        label="Goalkeeper"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=12,
                   facecolor=BG, edgecolor="#555", labelcolor=WHITE, bbox_to_anchor=(0.32, 0.006))
        fig.text(0.58, 0.018, "━━━▶  What they did", color=WHITE, fontsize=11, ha="left")
        fig.text(0.58, 0.010, "╌╌╌▶ ⭐ What they should have done (ML)", color=GOLD, fontsize=11, ha="left")
        fig.text(0.58, 0.002, "╌╌╌▶ ◆ Carry into space (ML)", color=LIME, fontsize=11, ha="left")
        fig.text(0.02, 0.006,
                 "* Recommendations powered by XGBoost model trained on 64 WC2022 matches. "
                 "360 data captures broadcast-visible players only.",
                 color="#666", fontsize=10, ha="left")

        out_path = players_dir / f"{safe_name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        summary_rows.append({
            "player": player_name,
            "team": team_name,
            "actions": total,
            "avg_dq": round(avg_dq, 3),
            "avg_delta_xt": round(p_df["delta_xt"].mean(), 6),
            "turnovers": turnovers,
            "turnover_pct": round(to_pct, 1),
            "goals_involved": goals,
        })

    # Save player summary
    summary_df = pd.DataFrame(summary_rows).sort_values("avg_dq", ascending=False)
    summary_df.to_csv(out_dir / "player_summary.csv", index=False)
    print(f"\nSaved player summary → {out_dir / 'player_summary.csv'}")

    print(f"\n✅ Done! All outputs → {out_dir}/")
    print(f"   - decision_quality.csv")
    print(f"   - player_summary.csv")
    print(f"   - players/  ({len(player_names)} reports)")


if __name__ == "__main__":
    main()
