#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Forecast Roblox Up-and-Coming chart top-10, place by place, over the
scored window read from task.json.

Model: damped-rank-momentum with mean reversion, blended with a player-growth
signal and an age-out hazard, converted to per-place probabilities with a
noisy-ranking (Gaussian around predicted place) softmax whose uncertainty
grows with lead time. Stdlib only, deterministic (no randomness used).
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
OUTPUT_NAME = "forecast.json"

MAX_PLACES_ROWS = 100


def _envf(name, default):
    """Optional tuning overrides; the grader sets none of these."""
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


TAU = _envf("M_TAU", 15.0)          # momentum decay scale (hours)
SIGMA0 = _envf("M_SIGMA0", 1.6)     # rank uncertainty now
SIGMA_G = _envf("M_SIGMA_G", 0.06)  # rank uncertainty growth per hour
UNNAMED_KEEP = _envf("M_KEEP", 0.94)  # share of each place left unnamed
SPREAD = _envf("M_SPREAD", 1.6)     # softmax spread over rank ordering


def parse_iso(s):
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def load_history(path):
    reads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("ok") is False:
                continue
            reads.append(o)
    reads.sort(key=lambda r: parse_iso(r["at"]))
    return reads


def game_rows(read_entry):
    """(rank, universe_id, playing) tuples of the up_and_coming chart."""
    out = []
    rows = read_entry.get("up_and_coming") or []
    for r in rows:
        if r.get("universe_id") is None or r.get("rank") is None:
            continue
        out.append((int(r["rank"]), int(r["universe_id"]), r.get("playing")))
    out.sort(key=lambda t: t[0])
    return out


def build_features(reads):
    """Per-universe trajectory features over the history."""
    prev_rank = {}
    prev_playing = {}
    prev_t = None
    vel_samples = {}   # uid -> list of (dt_hours, drank, play_rate)
    first_seen = {}
    last_seen = {}
    last_play = {}
    last_rank = {}
    created = {}

    for entry in reads:
        t = parse_iso(entry["at"])
        rows = game_rows(entry)
        uids = set()
        for rank, uid, playing in rows:
            uids.add(uid)
            if uid not in first_seen:
                first_seen[uid] = t
            last_seen[uid] = t
            if playing is not None:
                last_play[uid] = playing
            last_rank[uid] = rank
            if prev_t is not None:
                dt = (t - prev_t).total_seconds() / 3600.0
                if dt > 0:
                    pr = prev_rank.get(uid)
                    pp = prev_playing.get(uid)
                    dr = None
                    if pr is not None:
                        dr = rank - pr
                    dpr = None
                    if playing is not None and pp is not None:
                        dpr = (playing - pp) / dt
                    if dr is not None or dpr is not None:
                        vel_samples.setdefault(uid, []).append((dt, dr, dpr))
            prev_rank[uid] = rank
            prev_playing[uid] = playing
        # games that dropped off this read lose their player-count baseline
        for uid in list(prev_playing.keys()):
            if uid not in uids:
                prev_playing.pop(uid, None)
        prev_t = t
        for d in (entry.get("details") or []):
            if not d:
                continue
            uid = d.get("universe_id")
            if uid is not None and d.get("created"):
                created[int(uid)] = d["created"]

    # EWMA-style velocity per universe (per-hour rank velocity + play growth),
    # weighting short gaps more (recent reads matter more).
    vel = {}
    play_vel = {}
    for uid, samples in vel_samples.items():
        rv = 0.0
        pv = 0.0
        wsum = 0.0
        for dt, dr, dpr in samples:
            w = 1.0 / math.sqrt(max(dt, 0.5))
            if dr is not None:
                rv += w * (dr / dt)
            if dpr is not None:
                pv += w * dpr
            wsum += w
        if wsum > 0:
            vel[uid] = rv / wsum
            play_vel[uid] = pv / wsum
        else:
            vel[uid] = 0.0
            play_vel[uid] = 0.0

    t_end = parse_iso(reads[-1]["at"])
    feats = {}
    for uid in last_rank:
        feats[uid] = {
            "rank": last_rank[uid],
            "vel": vel.get(uid, 0.0),
            "play_vel": play_vel.get(uid, 0.0),
            "playing": last_play.get(uid) or 0,
            "age_hours_chart": (t_end - first_seen.get(uid, t_end)).total_seconds() / 3600.0,
            "created": created.get(uid),
        }
    return feats, t_end


def created_age_hours(feat, t_end):
    c = feat.get("created")
    if not c:
        return None
    try:
        return (t_end - parse_iso(c)).total_seconds() / 3600.0
    except Exception:
        return None


def predicted_rank(feat, h, max_rank, t_end):
    """Predicted (continuous) chart rank h hours after the history ends."""
    r0 = feat["rank"]
    vel = feat["vel"]  # positive = falling down the chart
    damped = vel * TAU * (1.0 - math.exp(-h / TAU))
    # growth signal: fast relative player growth implies climbing
    pv = feat["play_vel"]
    playing = max(feat["playing"], 1)
    growth = pv / playing  # relative hourly growth
    growth_pull = -min(max(growth, -0.3), 0.3) * 6.0 * min(h, 24.0) / 24.0
    r = r0 + damped + growth_pull
    # mild mean reversion: unknown newcomers push everyone down over time
    r += 0.01 * h
    # age-out hazard: games near the 28-day publish limit fall off
    ca = created_age_hours(feat, t_end)
    if ca is not None:
        days_left = (28.0 * 24.0 - ca) / 24.0
        if days_left <= 0:
            r = max_rank + 5.0
        elif days_left < 3.0:
            r += (3.0 - days_left) * 8.0
    # recent entrants keep improving slightly
    if feat["age_hours_chart"] < 24.0:
        r -= (24.0 - feat["age_hours_chart"]) / 24.0 * 1.5
    return max(0.5, min(max_rank + 5.0, r))


def place_probabilities(feats, uid_list, t_end, hours, places, max_rank):
    """{place: {uid: p}} for a given lead time in hours."""
    sigma = SIGMA0 + SIGMA_G * hours
    scores = {}
    for uid in uid_list:
        r = predicted_rank(feats[uid], hours, max_rank, t_end)
        if r > max_rank + 1.0:
            continue
        scores[uid] = r
    if not scores:
        return {k: {} for k in range(1, places + 1)}
    best = min(scores.values())
    weights = {u: math.exp(-(r - best) / (SPREAD * sigma)) for u, r in scores.items()}
    wsum = sum(weights.values())
    probs = {u: w / wsum for u, w in weights.items()}
    place_probs = {}
    for k in range(1, places + 1):
        row = {}
        z = 0.0
        for u, r in scores.items():
            p = probs[u] * math.exp(-((r - k) ** 2) / (2.0 * sigma * sigma))
            row[u] = p
            z += p
        if z > 0:
            scale = UNNAMED_KEEP / z  # leave the rest unnamed as insurance
            row = {u: min(v * scale, 0.98) for u, v in row.items()}
            row = {u: v for u, v in row.items() if v >= 0.002}
        place_probs[k] = row
    return place_probs


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    window = task["window"]
    slot_minutes = int(window.get("slot_minutes", 60))
    slots = int(window.get("slots", 72))

    reads = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not reads:
        raise SystemExit("empty history")
    feats, t_end = build_features(reads)
    max_rank = max((len(game_rows(r)) for r in reads), default=50)

    uid_list = sorted(feats.keys())
    agg = {k: {} for k in range(1, places + 1)}
    for s in range(slots):
        hours = (s + 1) * slot_minutes / 60.0
        pp = place_probabilities(feats, uid_list, t_end, hours, places, max_rank)
        for k in range(1, places + 1):
            for u, p in pp[k].items():
                agg[k][u] = agg[k].get(u, 0.0) + p / slots

    probabilities = []
    for k in range(1, places + 1):
        row = sorted(agg[k].items(), key=lambda kv: (-kv[1], kv[0]))
        total = 0.0
        for u, p in row[:MAX_PLACES_ROWS]:
            if total + p > 1.0:
                p = max(0.0, 1.0 - total)
            if p <= 0:
                break
            probabilities.append({"universe_id": int(u), "place": k, "p": round(p, 6)})
            total += p
            if total >= 0.999:
                break

    # top10: best mean predicted rank over the window, ties by uid
    mean_rank = {}
    for uid in uid_list:
        acc = 0.0
        for s in range(slots):
            hours = (s + 1) * slot_minutes / 60.0
            acc += predicted_rank(feats[uid], hours, max_rank, t_end)
        mean_rank[uid] = acc / slots
    top10 = sorted(uid_list, key=lambda u: (mean_rank[u], u))[:places]

    out = {"probabilities": probabilities, "top10": [int(u) for u in top10]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {path}: {len(probabilities)} probability rows, top10={out['top10']}")


if __name__ == "__main__":
    main()
