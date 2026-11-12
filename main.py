#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Forecast Roblox Up-and-Coming chart top-10, place by place.

Model: each game's rank follows a damped-momentum drift; at every hourly slot
of the scored window its place distribution is a discretized Gaussian around
the projected rank (variance growing with the horizon).  Probabilities are the
slot-average of those distributions, scaled to leave an honest unnamed share
at every place (new games we cannot name join the chart during the window).
"""

import json
import math
import os
from datetime import datetime, timedelta

# ---- model knobs -----------------------------------------------------------
MOMENTUM = 0.30        # fraction of recent per-hour rank velocity kept
DAMPING = 0.85         # velocity decays per simulated hour
SIGMA0 = 0.60          # rank noise (per sqrt hour) at the close
SIGMA_GROWTH = 0.08    # extra noise per sqrt hour as the horizon grows
TARGET_SUM = 0.15      # named mass per place; rest stays unnamed
MAX_ROWS = 25          # named candidates per place
P_FLOOR = 0.0005       # drop rows below this
VELOCITY_BLEND = 0.30  # blend of long-run and recent velocity estimate

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    reads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok", True):
                continue
            games = r.get("up_and_coming") or []
            if not games:
                continue
            reads.append({"at": parse_ts(r["at"]), "games": games})
    reads.sort(key=lambda x: x["at"])
    return reads


def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_velocities(reads):
    """Per-hour rank velocity per game: blend of a long-run estimate and the
    most recent interval, damped toward zero."""
    last = reads[-1]
    ranks = {g["universe_id"]: g["rank"] for g in last["games"]}
    vel = {uid: 0.0 for uid in ranks}
    n = len(reads)
    # long-run: rank change over the last ~24h of history
    cutoff = last["at"] - timedelta(hours=24)
    base = next((r for r in reads if r["at"] >= cutoff), None)
    long_run = {}
    if base is not None and base["at"] < last["at"]:
        dt = (last["at"] - base["at"]).total_seconds() / 3600.0
        r0 = {g["universe_id"]: g["rank"] for g in base["games"]}
        for uid, rk in ranks.items():
            if uid in r0:
                long_run[uid] = (r0[uid] - rk) / dt  # positive = climbing
    # recent: exponentially weighted over the last few intervals
    rec_w, rec_s = 0.0, {uid: 0.0 for uid in ranks}
    for back in range(1, min(4, n - 1) + 1):
        a, b = reads[n - 1 - back], reads[n - back]
        dt = (b["at"] - a["at"]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        ra = {g["universe_id"]: g["rank"] for g in a["games"]}
        w = 0.6 ** (back - 1)
        rec_w += w
        for uid, rk in ranks.items():
            if uid in ra:
                rec_s[uid] += w * (ra[uid] - rk) / dt
    recent = {uid: (rec_s[uid] / rec_w if rec_w else 0.0) for uid in ranks}
    for uid in ranks:
        lr = long_run.get(uid, 0.0)
        rc = recent.get(uid, 0.0)
        raw = VELOCITY_BLEND * lr + (1.0 - VELOCITY_BLEND) * rc
        # shrink extreme velocities: they rarely persist
        vel[uid] = MOMENTUM * math.tanh(raw / 2.0) * 2.0
    return ranks, vel


def place_probs(mu, sigma, max_place=60):
    """Discretized Gaussian over places 1..max_place, normalized."""
    K = range(1, max_place + 1)
    raw = [math.exp(-0.5 * ((k - mu) / sigma) ** 2) for k in K]
    s = sum(raw)
    if s <= 0:
        p = [0.0] * max_place
        idx = min(max(int(round(mu)) - 1, 0), max_place - 1)
        p[idx] = 1.0
        return p
    return [x / s for x in raw]


def forecast(reads, task):
    places = task.get("places", 10)
    window = task["window"]
    slots = window.get("slots", 72)
    slot_min = window.get("slot_minutes", 60)
    n_hours = slots * slot_min // 60

    ranks, vel = build_velocities(reads)
    max_place = min(60, max(ranks.values()) + 5)

    # accumulate slot-averaged place distributions
    agg = {uid: [0.0] * max_place for uid in ranks}
    for h in range(1, n_hours + 1):
        for uid, rk in ranks.items():
            v = vel[uid]
            mu = rk - v * DAMPING * (1.0 - DAMPING ** h) / (1.0 - DAMPING)
            sigma = math.sqrt(SIGMA0 ** 2 + (SIGMA_GROWTH ** 2) * h)
            p = place_probs(mu, sigma, max_place)
            a = agg[uid]
            for k in range(max_place):
                a[k] += p[k]
    total = n_hours
    for uid in agg:
        agg[uid] = [x / total for x in agg[uid]]

    # build rows, then scale each place down to TARGET_SUM
    by_place = {k + 1: [] for k in range(max_place)}
    for uid, cs in agg.items():
        for k in range(places):
            if cs[k] >= P_FLOOR:
                by_place[k + 1].append({"universe_id": uid, "place": k + 1,
                                        "p": cs[k]})
    probabilities = []
    for plc in range(1, places + 1):
        rows = sorted(by_place[plc], key=lambda r: -r["p"])[:MAX_ROWS]
        ssum = sum(r["p"] for r in rows)
        scale = min(TARGET_SUM / ssum, 1.0) if ssum > 0 else 0.0
        for r in rows:
            r["p"] = round(r["p"] * scale, 6)
        rows = [r for r in rows if r["p"] >= P_FLOOR]
        probabilities.extend(rows)

    # top10: expected place over the window (never-placed = current rank)
    mean_place = {}
    for uid, cs in agg.items():
        s_placed = sum(cs[:places])
        mean_place[uid] = (sum((k + 1) * c for k, c in enumerate(cs[:places]))
                           + (1.0 - s_placed) * ranks.get(uid, 50.0))
    top10 = [uid for uid, _ in sorted(mean_place.items(),
                                      key=lambda kv: kv[1])[:places]]
    return {"probabilities": probabilities, "top10": top10}


def main():
    history_path = os.path.join(INPUT_DIR, "history.ndjson")
    task_path = os.path.join(INPUT_DIR, "task.json")
    reads = load_history(history_path)
    if not reads:
        raise SystemExit("no valid reads in history")
    task = load_task(task_path) if os.path.exists(task_path) else {
        "places": 10,
        "window": {"from": reads[-1]["at"].isoformat(),
                   "slots": 72, "slot_minutes": 60},
    }
    result = forecast(reads, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"wrote {out_path}: {len(result['probabilities'])} rows")


if __name__ == "__main__":
    main()
