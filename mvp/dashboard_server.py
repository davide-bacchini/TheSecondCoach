"""
dashboard_server.py — Flask API for Football Analytics Manager Dashboard
=========================================================================
Wraps the analyze_match.py pipeline and serves JSON + static frontend.

Usage:
  source .venv/bin/activate
  python dashboard_server.py
  → Open http://localhost:5050
"""

import json, math, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS

# Import core functions from analyze_match
from analyze_match import (
    process_match_events, generate_candidates, score_candidates,
    read_json, xT_lookup, pdist, short_name, make_safe_name,
    FEATURES, PITCH_X, PITCH_Y, XT_COLS, XT_ROWS, GOAL_X, GOAL_Y,
    needs_mirror, mirror_frame, angle_to_goal, point_to_line_dist, clamp,
)

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def json_response(data, status=200):
    """Return a Flask Response with JSON, handling numpy types."""
    return Response(
        json.dumps(data, cls=NumpyEncoder, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "vaep_xgb.json")
XT_GRID_PATH = str(Path(__file__).resolve().parent.parent / "models" / "xt_grid.csv")

# ── Lazy-load model and grid ────────────────────────────────────────────────
_model = None
_xt_grid = None


def get_model():
    global _model
    if _model is None:
        from xgboost import XGBRegressor
        _model = XGBRegressor()
        _model.load_model(MODEL_PATH)
    return _model


def get_xt_grid():
    global _xt_grid
    if _xt_grid is None:
        _xt_grid = pd.read_csv(XT_GRID_PATH, index_col=0).values
    return _xt_grid


# ═══════════════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/matches")
def list_matches():
    """List all available matches across all competitions with 360 data."""
    comps = json.loads((DATA_DIR / "competitions.json").read_text())
    matches_out = []

    for c in comps:
        cid, sid = c["competition_id"], c["season_id"]
        mp = DATA_DIR / "matches" / str(cid) / f"{sid}.json"
        if not mp.exists():
            continue
        matches = json.loads(mp.read_text())
        for m in matches:
            mid = m["match_id"]
            has_360 = (DATA_DIR / "three-sixty" / f"{mid}.json").exists()
            has_events = (DATA_DIR / "events" / f"{mid}.json").exists()
            if has_360 and has_events:
                matches_out.append({
                    "match_id": mid,
                    "comp_id": cid,
                    "season_id": sid,
                    "competition": c.get("competition_name", ""),
                    "season": c.get("season_name", ""),
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                    "home_score": m.get("home_score", 0),
                    "away_score": m.get("away_score", 0),
                    "match_date": m.get("match_date", ""),
                    "stadium": m.get("stadium", {}).get("name", ""),
                })

    matches_out.sort(key=lambda x: x["match_date"], reverse=True)
    return json_response({"matches": matches_out, "total": len(matches_out)})


@app.route("/api/analyze", methods=["POST"])
def analyze_match_api():
    """Process a match through the full ML pipeline and return JSON."""
    data = request.json
    match_id = int(data["match_id"])
    comp_id = int(data["comp_id"])
    season_id = int(data["season_id"])
    min_actions = int(data.get("min_actions", 15))

    model = get_model()
    xt_grid = get_xt_grid()

    # ── Load match data ──────────────────────────────────────────────────────
    matches = read_json(DATA_DIR / "matches" / str(comp_id) / f"{season_id}.json")
    m_info = next(m for m in matches if int(m["match_id"]) == match_id)

    home = m_info["home_team"]["home_team_name"]
    away = m_info["away_team"]["away_team_name"]
    hs = int(m_info.get("home_score", 0) or 0)
    ask = int(m_info.get("away_score", 0) or 0)
    match_meta = {
        "match_id": match_id,
        "home_team": home, "away_team": away,
        "home_score": hs, "away_score": ask,
        "match_date": m_info.get("match_date", ""),
        "competition": m_info.get("competition", {}).get("competition_name", ""),
        "season": m_info.get("season", {}).get("season_name", ""),
        "stadium": m_info.get("stadium", {}).get("name", ""),
    }

    events = read_json(DATA_DIR / "events" / f"{match_id}.json")
    frames_list = read_json(DATA_DIR / "three-sixty" / f"{match_id}.json")
    frames_map = {f["event_uuid"]: f for f in frames_list}

    # ── Extract player positions from Starting XI + Substitutions ────────────
    player_positions = {}  # player_name → {position, role}
    ROLE_MAP = {
        "Goalkeeper": "GK",
    }
    DEF_KEYWORDS = ("Back", "Center Back", "Wing Back")
    MID_KEYWORDS = ("Midfield",)
    FWD_KEYWORDS = ("Wing", "Forward", "Striker")

    def classify_role(pos_name):
        if "Goalkeeper" in pos_name:
            return "GK"
        for kw in DEF_KEYWORDS:
            if kw in pos_name and "Midfield" not in pos_name:
                return "DEF"
        for kw in FWD_KEYWORDS:
            if kw in pos_name and "Midfield" not in pos_name:
                return "FWD"
        for kw in MID_KEYWORDS:
            if kw in pos_name:
                return "MID"
        return "MID"  # fallback

    for e in events:
        if e["type"]["name"] == "Starting XI":
            for p in e.get("tactics", {}).get("lineup", []):
                pname = p["player"]["name"]
                pos = p["position"]["name"]
                player_positions[pname] = {
                    "position": pos,
                    "role": classify_role(pos),
                }
        elif e["type"]["name"] == "Substitution":
            # The substituted-on player inherits the position
            sub_on = e.get("substitution", {}).get("replacement", {}).get("name")
            pos_name = e.get("position", {}).get("name", "")
            if sub_on and pos_name:
                player_positions[sub_on] = {
                    "position": pos_name,
                    "role": classify_role(pos_name),
                }

    # ── Process events ───────────────────────────────────────────────────────
    df = process_match_events(events, frames_map, xt_grid, match_meta)

    # ── Build player data ────────────────────────────────────────────────────
    players_data = []
    for player_name, p_df in df.groupby("player"):
        if len(p_df) < min_actions:
            continue

        team = p_df["team"].iloc[0]
        pos_info = player_positions.get(player_name, {"position": "Unknown", "role": "MID"})
        position = pos_info["position"]
        role = pos_info["role"]
        total = len(p_df)
        turnovers = int(p_df["is_turnover"].sum())
        avg_dq = round(float(p_df["decision_quality"].mean()), 3)
        avg_dxt = round(float(p_df["delta_xt"].mean()), 6)
        goals = int(p_df["goal_within_5_actions"].sum())
        passes = int((p_df["event_type"] == "Pass").sum())
        completed_passes = int(
            ((p_df["event_type"] == "Pass") & (~p_df["is_turnover"])).sum()
        )
        carries = int((p_df["event_type"] == "Carry").sum())
        shots = int((p_df["event_type"] == "Shot").sum())
        goals_scored = int(
            ((p_df["event_type"] == "Shot") &
             (p_df["outcome"].str.contains("Goal", na=False))).sum()
        )
        dribbles = int((p_df["event_type"] == "Dribble").sum())

        # Under pressure %
        pressure_pct = round(float(p_df["under_pressure"].mean()) * 100, 1)
        pass_accuracy = round(completed_passes / max(passes, 1) * 100, 1)

        # Progressive actions (positive delta_xt)
        progressive = int((p_df["delta_xt"] > 0.005).sum())

        # Select best/worst actions with 360 data
        has_frame_mask = p_df["event_uuid"].isin(frames_map)
        is_goal_mask = p_df["goal_within_5_actions"].astype(bool)
        valid_mask = has_frame_mask | is_goal_mask
        valid_df = p_df[valid_mask]
        pool = valid_df if len(valid_df) >= 10 else p_df

        best_5 = pool.nlargest(5, "delta_xt")
        worst_5 = pool.nsmallest(5, "decision_quality")

        def action_to_json(row, is_best):
            """Convert a single action row to JSON with freeze frame + ML rec."""
            euuid = row["event_uuid"]
            frame = frames_map.get(euuid)
            sx, sy = float(row["start_x"]), float(row["start_y"])

            # Fix mirroring
            if frame and needs_mirror(row["event_type"], frame, sx, sy):
                frame = mirror_frame(frame)
                frames_map[euuid] = frame

            # Freeze frame data for client rendering
            freeze_frame = []
            if frame:
                for p in frame.get("freeze_frame", []):
                    if not p.get("location"):
                        continue
                    freeze_frame.append({
                        "x": clamp(p["location"][0], 0.5, 119.5),
                        "y": clamp(p["location"][1], 0.5, 79.5),
                        "teammate": bool(p.get("teammate")),
                        "actor": bool(p.get("actor")),
                        "keeper": bool(p.get("keeper", False)),
                    })

            # ML recommendation for worst actions
            recommendation = None
            if not is_best and frame:
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
                    sx, sy, frame, spatial,
                    row.get("score_diff", 0),
                    row.get("under_pressure", False),
                    xt_grid,
                )
                best_cand = score_candidates(candidates, model) if candidates else None

                if best_cand:
                    btx, bty = best_cand["target_x"], best_cand["target_y"]
                    btype = best_cand["action_label"]
                    pred = best_cand["predicted_delta_xT"]

                    ex_val = float(row["end_x"]) if pd.notna(row.get("end_x")) else None
                    ey_val = float(row["end_y"]) if pd.notna(row.get("end_y")) else None

                    same_as_actual = False
                    if ex_val is not None and ey_val is not None:
                        actual_type = ("pass" if row["event_type"] == "Pass" else
                                       "carry" if row["event_type"] == "Carry" else "shot")
                        same_as_actual = (pdist(btx, bty, ex_val, ey_val) < 8.0
                                          and btype == actual_type)

                    if same_as_actual:
                        recommendation = {
                            "type": "execution",
                            "text": "Right idea, poor execution — timing or weight was off",
                            "target_x": btx, "target_y": bty,
                            "action_type": btype,
                            "predicted_dxt": round(pred, 4),
                        }
                    else:
                        direction = ("forward" if btx > sx + 5 else
                                     ("backward" if btx < sx - 5 else "wide"))
                        n_open = int(row.get("n_open_teammates", 0))

                        if btype == "shot":
                            text = f"Should have shot! ML ΔxT: {pred:+.4f}"
                        elif btype == "carry":
                            text = f"Carry {direction} into space. ML ΔxT: {pred:+.4f}"
                        else:
                            text = (f"Should have passed {direction}"
                                    f" ({n_open} open teammates). ML ΔxT: {pred:+.4f}")

                        recommendation = {
                            "type": btype,
                            "text": text,
                            "target_x": round(btx, 1),
                            "target_y": round(bty, 1),
                            "action_type": btype,
                            "predicted_dxt": round(pred, 4),
                        }

            end_x = float(row["end_x"]) if pd.notna(row.get("end_x")) else None
            end_y = float(row["end_y"]) if pd.notna(row.get("end_y")) else None

            return {
                "event_uuid": euuid,
                "event_type": row["event_type"],
                "outcome": str(row.get("outcome", "")),
                "minute": int(row.get("minute", 0)),
                "second": int(row.get("second", 0)),
                "start_x": round(sx, 1),
                "start_y": round(sy, 1),
                "end_x": round(end_x, 1) if end_x else None,
                "end_y": round(end_y, 1) if end_y else None,
                "delta_xt": round(float(row["delta_xt"]), 4),
                "decision_quality": round(float(row["decision_quality"]), 3),
                "is_turnover": bool(row.get("is_turnover")),
                "under_pressure": bool(row.get("under_pressure")),
                "goal_within_5": bool(row.get("goal_within_5_actions")),
                "freeze_frame": freeze_frame,
                "recommendation": recommendation,
            }

        best_actions = [action_to_json(row, True) for _, row in best_5.iterrows()]
        worst_actions = [action_to_json(row, False) for _, row in worst_5.iterrows()]

        # ── Generate position-aware coaching report ────────────────────────
        coaching_lines = []
        to_pct = round(turnovers / max(total, 1) * 100, 0)

        # ML recommendation patterns from worst actions
        rec_types = {}
        for wa in worst_actions:
            if wa.get("recommendation") and wa["recommendation"].get("action_type"):
                atype = wa["recommendation"]["action_type"]
                rec_types[atype] = rec_types.get(atype, 0) + 1

        # ── Position-specific intro and context ──
        if role == "GK":
            coaching_lines.append(
                f"🧤 Goalkeeper — primary role is distribution and shot-stopping. "
                f"Evaluated on pass accuracy and distribution under pressure."
            )
            if pass_accuracy >= 90:
                coaching_lines.append(
                    f"Strong distribution — {pass_accuracy}% pass accuracy. "
                    f"Confident playing out from the back."
                )
            elif pass_accuracy >= 75:
                coaching_lines.append(
                    f"Adequate distribution ({pass_accuracy}% accuracy) — "
                    f"some passes under pressure went astray."
                )
            else:
                coaching_lines.append(
                    f"⚠️ Distribution under pressure is a concern — "
                    f"only {pass_accuracy}% pass accuracy. "
                    f"Consider opting for longer clearances rather than risky short passes."
                )
            if turnovers > 0:
                coaching_lines.append(
                    f"Lost possession {turnovers} times ({to_pct:.0f}%). "
                    f"For a goalkeeper, turnovers can be catastrophic — "
                    f"prioritize safety over progressive distribution."
                )

        elif role == "DEF":
            coaching_lines.append(
                f"🛡️ Defender ({position}) — evaluated on positional discipline, "
                f"ball retention, and progressive passing."
            )
            if pass_accuracy >= 90:
                coaching_lines.append(
                    f"Excellent ball retention — {pass_accuracy}% pass accuracy, "
                    f"solid in build-up play."
                )
            if turnovers > 0:
                if to_pct > 10:
                    coaching_lines.append(
                        f"⚠️ {turnovers} turnovers ({to_pct:.0f}%) is too high for a defender. "
                        f"Losing the ball in defensive areas creates immediate danger."
                    )
                else:
                    coaching_lines.append(
                        f"{turnovers} turnovers ({to_pct:.0f}%) — solid ball security."
                    )
            if progressive > total * 0.3:
                coaching_lines.append(
                    f"Progressive passer — {progressive}/{total} actions moved the ball forward. "
                    f"Good at breaking lines."
                )

        elif role == "MID":
            coaching_lines.append(
                f"🎯 Midfielder ({position}) — evaluated on ball progression, "
                f"creativity, and decision-making under pressure."
            )
            best_types = best_5["event_type"].value_counts()
            if "Carry" in best_types and best_types.get("Carry", 0) >= 2:
                coaching_lines.append(
                    f"Strong ball carrier — {best_types['Carry']} of top 5 actions were carries. "
                    f"Effective at driving forward through midfield."
                )
            if "Pass" in best_types and best_types.get("Pass", 0) >= 3:
                coaching_lines.append(
                    f"Excellent passer — {best_types['Pass']} of top 5 actions were passes. "
                    f"Good at finding teammates in dangerous positions."
                )
            if turnovers > 0:
                coaching_lines.append(
                    f"Lost the ball {turnovers} times ({to_pct:.0f}%). "
                    f"{'Acceptable for an attacking mid.' if 'Attacking' in position else 'Should aim to reduce this.'}"
                )
            if progressive > total * 0.4:
                coaching_lines.append(
                    f"Highly progressive — {progressive}/{total} actions moved the ball forward."
                )

        elif role == "FWD":
            coaching_lines.append(
                f"⚡ Attacker ({position}) — evaluated on finishing, "
                f"chance creation, and off-the-ball movement."
            )
            if goals_scored > 0:
                coaching_lines.append(
                    f"Clinical finisher — scored {goals_scored} goal{'s' if goals_scored > 1 else ''}."
                )
            elif shots > 0:
                coaching_lines.append(
                    f"Took {shots} shot{'s' if shots > 1 else ''} without scoring. "
                    f"Shot quality and placement need attention."
                )
            if turnovers > 0:
                if to_pct <= 20:
                    coaching_lines.append(
                        f"{turnovers} turnovers ({to_pct:.0f}%) — acceptable for an attacker "
                        f"who takes risks in the final third."
                    )
                else:
                    coaching_lines.append(
                        f"⚠️ {turnovers} turnovers ({to_pct:.0f}%) is excessive even for an attacker. "
                        f"Need to be more selective about when to take on opponents."
                    )
            if dribbles > 0:
                coaching_lines.append(
                    f"Attempted {dribbles} dribble{'s' if dribbles > 1 else ''} — "
                    f"creativity in 1v1 situations."
                )

        # ── ML recommendation patterns (position-aware) ──
        if rec_types.get("shot", 0) >= 2:
            if role in ("FWD", "MID"):
                coaching_lines.append(
                    f"In {rec_types['shot']} of 5 worst decisions, the model says shooting "
                    f"was the better option. For a{'n attacker' if role == 'FWD' else ' midfielder'}, "
                    f"this suggests being too hesitant in front of goal."
                )
            else:
                coaching_lines.append(
                    f"In {rec_types['shot']} of 5 worst decisions, the model recommended a shot — "
                    f"unusual for a {position.lower()}, but suggests valuable "
                    f"opportunities were missed from advanced positions."
                )
        if rec_types.get("carry", 0) >= 2:
            coaching_lines.append(
                f"In {rec_types['carry']} of 5 worst decisions, carrying into space "
                f"was the better option. {'Look for space before committing to a pass.' if role != 'GK' else 'Rarely applicable for a GK.'}"
            )
        if rec_types.get("pass", 0) >= 2:
            if role == "GK":
                coaching_lines.append(
                    f"In {rec_types['pass']} of 5 worst decisions, a different distribution "
                    f"option was available. Focus on scanning before distributing."
                )
            else:
                coaching_lines.append(
                    f"In {rec_types['pass']} of 5 worst decisions, a different passing "
                    f"option was available."
                )

        # ── Position-aware actionable recommendation ──
        if role == "GK":
            if to_pct > 15:
                coaching_lines.append(
                    "💡 Recommendation: Simplify distribution — play shorter, safer passes "
                    "to nearby defenders rather than attempting risky long balls under pressure."
                )
            elif pass_accuracy < 80:
                coaching_lines.append(
                    "💡 Recommendation: Work on distribution accuracy. As a modern GK, "
                    "accurate short passing is essential for build-up play."
                )
            else:
                coaching_lines.append(
                    "💡 Solid distribution performance. Continue commanding the build-up."
                )
        elif role == "DEF":
            if to_pct > 12:
                coaching_lines.append(
                    "💡 Recommendation: Reduce turnovers in your own half — "
                    "opt for the safe option when pressed, and clear when in doubt."
                )
            elif rec_types.get("carry", 0) >= 2:
                coaching_lines.append(
                    "💡 Recommendation: When space opens in front, carry forward "
                    "to draw opponents and create numerical advantages in midfield."
                )
            elif avg_dq >= 0.80:
                coaching_lines.append(
                    "💡 Strong defensive performance. Reliable in possession."
                )
            else:
                coaching_lines.append(
                    "💡 Recommendation: Focus on keeping possession simple. "
                    "A defender's first job is not to lose the ball."
                )
        elif role == "MID":
            if rec_types.get("shot", 0) >= 2:
                coaching_lines.append(
                    "💡 Recommendation: Be more decisive in the final third — "
                    "shoot when the angle is there instead of looking for the extra pass."
                )
            elif to_pct > 15:
                coaching_lines.append(
                    "💡 Recommendation: Reduce risk in transitions — "
                    "use simple passes to keep the team's rhythm."
                )
            elif avg_dq >= 0.80:
                coaching_lines.append(
                    "💡 Excellent midfield performance. Dictating play well."
                )
            else:
                coaching_lines.append(
                    "💡 Recommendation: Look for forward passing options more often — "
                    "progressive passes create more value."
                )
        elif role == "FWD":
            if rec_types.get("shot", 0) >= 2:
                coaching_lines.append(
                    "💡 Recommendation: Take earlier shots when in the box. "
                    "Don't over-dribble — the first touch in the box should be a shot."
                )
            elif to_pct > 25:
                coaching_lines.append(
                    "💡 Recommendation: Be more selective about when to dribble. "
                    "Release the ball quicker to teammates in better positions."
                )
            elif goals_scored > 0 and avg_dq >= 0.75:
                coaching_lines.append(
                    "💡 Strong attacking display. Clinical when it mattered."
                )
            else:
                coaching_lines.append(
                    "💡 Recommendation: Make more direct runs and get into shooting positions. "
                    "Movement off the ball is key to creating chances."
                )

        players_data.append({
            "name": player_name,
            "short_name": short_name(player_name),
            "team": team,
            "position": position,
            "role": role,
            "actions": total,
            "avg_dq": avg_dq,
            "avg_delta_xt": avg_dxt,
            "goals_scored": goals_scored,
            "goals_involved": goals,
            "shots": shots,
            "passes": passes,
            "completed_passes": completed_passes,
            "pass_accuracy": round(completed_passes / max(passes, 1) * 100, 1),
            "carries": carries,
            "dribbles": dribbles,
            "turnovers": turnovers,
            "turnover_pct": round(turnovers / max(total, 1) * 100, 1),
            "progressive_actions": progressive,
            "pressure_pct": pressure_pct,
            "best_actions": best_actions,
            "worst_actions": worst_actions,
            "coaching_report": coaching_lines,
        })

    # Sort by decision quality
    players_data.sort(key=lambda x: -x["avg_dq"])

    # Team summaries
    teams = {}
    for p in players_data:
        t = p["team"]
        if t not in teams:
            teams[t] = {"name": t, "players": 0, "avg_dq": 0, "total_goals": 0,
                         "total_turnovers": 0, "total_actions": 0}
        teams[t]["players"] += 1
        teams[t]["avg_dq"] += p["avg_dq"]
        teams[t]["total_goals"] += p["goals_scored"]
        teams[t]["total_turnovers"] += p["turnovers"]
        teams[t]["total_actions"] += p["actions"]

    for t in teams.values():
        t["avg_dq"] = round(t["avg_dq"] / max(t["players"], 1), 3)

    return json_response({
        "match": match_meta,
        "teams": list(teams.values()),
        "players": players_data,
        "total_actions": int(len(df)),
        "total_players": len(players_data),
    })


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🏟️  Football Analytics Dashboard")
    print("   Loading model ...")
    get_model()
    get_xt_grid()
    print("   ✅ Ready!")
    print("   → Open http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
