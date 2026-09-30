"""Top-N next-location prediction from an id's place-visit sequence.

Sequences are per-id place visits ordered by time with consecutive repeats
collapsed. Models (all with backoff to lower orders, then to visit frequency):
* most_frequent  order-0: the id's most visited places.
* markov1        P(next | current place).
* markov2        P(next | previous, current).
* markov1_time   P(next | current place, time bucket) where the bucket is
                 weekday/weekend x 4-hour block of the *arrival* at the current place
                 (sequences carry start times only) in local time.
``evaluate`` trains on each id's first (1 - test_frac) of visits and scores every
test step with the true history as context: acc@k and mean reciprocal rank.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import polars as pl

from dwells import schema

MODELS = ("most_frequent", "markov1", "markov2", "markov1_time")


def _collapse(seq: pl.DataFrame) -> pl.DataFrame:
    s = seq.sort("id", "start_ts")
    keep = (pl.col("place_id") != pl.col("place_id").shift(1).over("id")).fill_null(True)
    return s.filter(keep)


def _bucket(ts, utc_offset_h):
    import datetime as _dt

    t = ts + _dt.timedelta(hours=utc_offset_h)
    return (t.weekday() >= 5) * 6 + t.hour // 4


class Predictor:
    def __init__(self, model: str, utc_offset_h: float = 0.0):
        self.model, self.off = model, utc_offset_h
        self.freq = defaultdict(Counter)
        self.c1 = defaultdict(Counter)
        self.c2 = defaultdict(Counter)
        self.ct = defaultdict(Counter)

    def update(self, uid, places, times):
        for i, p in enumerate(places):
            self.freq[uid][p] += 1
            if i >= 1:
                self.c1[(uid, places[i - 1])][p] += 1
                self.ct[(uid, places[i - 1], _bucket(times[i - 1], self.off))][p] += 1
            if i >= 2:
                self.c2[(uid, places[i - 2], places[i - 1])][p] += 1

    def rank(self, uid, hist, times) -> list:
        """Places ordered by predicted probability, given the visit history."""
        return [p for p, _ in self.ranked(uid, hist, times)]

    def ranked(self, uid, hist, times) -> list:
        """(place, probability) pairs; the probability comes from the most specific
        context table that proposes the place (backoff tables fill the rest)."""
        tables = []
        if self.model == "markov2" and len(hist) >= 2:
            tables.append(self.c2.get((uid, hist[-2], hist[-1])))
        if self.model == "markov1_time" and hist:
            tables.append(self.ct.get((uid, hist[-1], _bucket(times[-1], self.off))))
        if self.model != "most_frequent" and hist:
            tables.append(self.c1.get((uid, hist[-1])))
        tables.append(self.freq.get(uid))
        out, seen = [], set()
        for t in tables:
            if not t:
                continue
            tot = sum(v for p, v in t.items() if not hist or p != hist[-1]) or 1
            for p, c in t.most_common():
                if p not in seen and (not hist or p != hist[-1]):
                    seen.add(p)
                    out.append((p, c / tot))
        return out

    def predict_top(self, seq: pl.DataFrame, n: int = 3) -> pl.DataFrame:
        rows = []
        for (uid,), g in _collapse(seq).group_by("id", maintain_order=True):
            places = g["place_id"].to_list()
            times = g["start_ts"].to_list()
            for k, (p, prob) in enumerate(self.ranked(uid, places, times)[:n]):
                rows.append((uid, places[-1], k + 1, p, prob))
        return pl.DataFrame(rows, schema=schema.PREDICTIONS, orient="row")


def fit(seq: pl.DataFrame, model: str, utc_offset_h: float = 0.0) -> Predictor:
    m = Predictor(model, utc_offset_h)
    for (uid,), g in _collapse(seq).group_by("id", maintain_order=True):
        m.update(uid, g["place_id"].to_list(), g["start_ts"].to_list())
    return m


def evaluate(seq: pl.DataFrame, model: str, test_frac: float = 0.3, top_n=(1, 3, 5),
             utc_offset_h: float = 0.0) -> dict:
    m = Predictor(model, utc_offset_h)
    tests = []
    for (uid,), g in _collapse(seq).group_by("id", maintain_order=True):
        places, times = g["place_id"].to_list(), g["start_ts"].to_list()
        cut = int(round(len(places) * (1 - test_frac)))
        if cut < 2 or cut >= len(places):
            continue
        m.update(uid, places[:cut], times[:cut])
        tests.append((uid, places, times, cut))
    ranks = []
    for uid, places, times, cut in tests:
        for i in range(cut, len(places)):
            r = m.rank(uid, places[:i], times[:i])
            ranks.append(r.index(places[i]) + 1 if places[i] in r else np.inf)
    ranks = np.array(ranks, float)
    res = {f"acc@{k}": float(np.mean(ranks <= k)) if len(ranks) else 0.0 for k in top_n}
    res["mrr"] = float(np.mean(1 / ranks)) if len(ranks) else 0.0
    res["n_test"] = len(ranks)
    return res
