#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Backtest in-process: apply param overrides to main module, forecast on a
train split, house-score against later reads."""
import json
import math
import os
import sys

KIT = "/tmp/makefun/setup/roblox-up-and-coming-forecast-01-kit"
sys.path.insert(0, "/tmp/makefun/roblox-entry")
import main as M  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def score_forecast(fc, reads):
    probs = {}
    named = {}
    for r in fc["probabilities"]:
        probs.setdefault(r["place"], {})[r["universe_id"]] = r["p"]
    for plc in range(1, 11):
        m = probs.get(plc, {})
        named[plc] = (min(sum(m.values()), 1.0), set(m.keys()))
    scores, hits, agree = [], [], []
    for read in reads:
        chart_map = {g["universe_id"]: g["rank"] for g in read["up_and_coming"]}
        s = 0.0
        for plc in range(1, 11):
            tot, nm = named[plc]
            pool = [u for u in chart_map if u not in nm]
            share = (1.0 - tot) / len(pool) if pool else 0.0
            true_uid = next((u for u, rk in chart_map.items() if rk == plc), None)
            p = probs.get(plc, {}).get(true_uid, share if true_uid not in nm else 0.0)
            s += math.log(max(p, 0.0001))
        scores.append(s)
        t10 = [u for u, rk in sorted(chart_map.items(), key=lambda kv: kv[1])[:10]]
        hits.append(len(set(fc["top10"]) & set(t10)))
        my = {u: i for i, u in enumerate(fc["top10"])}
        agree.append(sum(10 - abs(my[u] - t10.index(u)) for u in t10 if u in my))
    return sum(scores) / len(scores), sum(hits) / len(hits), sum(agree) / len(agree), len(scores)


def run(overrides, split=0.7):
    for k, v in overrides.items():
        setattr(M, k, v)
    rows = load(os.path.join(KIT, "input", "history.ndjson"))
    task = json.load(open(os.path.join(KIT, "input", "task.json")))
    cut = int(len(rows) * split)
    train, test = rows[:cut], rows[cut:]
    from datetime import timedelta
    reads = [{"at": parse_ts(r["at"]), "games": r["up_and_coming"]} for r in train]
    last = parse_ts(train[-1]["at"])
    t2 = dict(task)
    t2["window"] = {"from": train[-1]["at"], "until": (last + timedelta(hours=72)).isoformat(),
                    "slot_minutes": 60, "slots": 72}
    fc = M.forecast(reads, t2)
    ls, hits, ag, n = score_forecast(fc, [{"up_and_coming": r["up_and_coming"]} for r in test])
    return ls, hits, ag, n, fc


def baseline_forecast(reads):
    """House starter baseline: near-persistence with neighbor mass."""
    last = {g["universe_id"]: g["rank"] for g in reads[-1]["games"]}
    order = sorted(last, key=lambda u: last[u])
    probs = []
    at = {u: i for i, u in enumerate(order)}
    for u in order[:11]:
        k = at[u]
        if k == 0:
            probs.append((u, 1, 0.5))
        elif k == 1:
            probs.append((u, 1, 0.3)); probs.append((u, 2, 0.5))
        else:
            if k <= 10:
                probs.append((u, k + 1, 0.5))
            if k - 1 >= 0 and k - 1 < 10:
                probs.append((u, k, 0.15))
            if k + 1 < 10:
                probs.append((u, k + 2, 0.15))
    agg = {}
    for u, plc, p in probs:
        agg.setdefault(plc, {})
        agg[plc][u] = agg[plc].get(u, 0.0) + p
    rows = [{"universe_id": u, "place": plc, "p": p}
            for plc, m in agg.items() for u, p in m.items()]
    return {"probabilities": rows, "top10": order[:10]}


if __name__ == "__main__":
    overrides = {}
    for kv in sys.argv[1:]:
        k, v = kv.split("=")
        try:
            overrides[k] = int(v) if k in ("SIMS", "MAX_ROWS") else float(v)
        except ValueError:
            overrides[k] = v
    ls, hits, ag, n, fc = run(overrides)
    print(f"ov={overrides or 'default'}")
    print(f"log_score={ls:.6f} typical_p={math.exp(ls/10):.5f} hits={hits:.2f} agree={ag:.1f} reads={n}")
    # baseline reference on the same split
    rows = load(os.path.join(KIT, "input", "history.ndjson"))
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    bls, bh, bag, _ = score_forecast(baseline_forecast(
        [{"at": parse_ts(r["at"]), "games": r["up_and_coming"]} for r in train]),
        [{"up_and_coming": r["up_and_coming"]} for r in test])
    print(f"baseline: log_score={bls:.6f} hits={bh:.2f} agree={bag:.1f}")
