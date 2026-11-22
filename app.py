"""Fight Intelligence Analyzer - single-file version. Run: streamlit run app.py"""

# ========== config ==========
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FIA_DATA_DIR", Path(__file__).resolve().parent / "data"))
SEED_FLAG = DATA_DIR / ".seeded"

# ========== schema ==========
STRIKE_CATS = ["punch", "kick", "knee", "elbow"]
ALL_CATS = STRIKE_CATS + ["clinch", "defence", "unclassified"]

TECHNIQUES = {
    "punch": ["jab", "cross", "hook", "uppercut", "overhand"],
    "kick": ["low kick", "body kick", "head kick", "teep"],
    "knee": ["straight knee", "diagonal knee", "clinch knee"],
    "elbow": ["horizontal elbow", "upward elbow", "spinning elbow"],
    "clinch": ["clinch entry", "clinch control", "sweep/throw"],
    "defence": ["check", "block", "parry", "slip", "evade", "kick catch"],
}
ALL_TECHS = [t for v in TECHNIQUES.values() for t in v]

LANDED = ["unknown", "yes", "no"]
RANGES = ["unknown", "long", "mid", "close"]
TRAINING_TYPES = ["pad work", "sparring"]

# Provenance of each record. Shown everywhere so auto data is never confused with manual data.
SRC_AUTO = "auto-detected"               # timing found by computer vision, no label yet
SRC_AUTO_LABELLED = "auto+manual label"  # CV timing, technique labelled by a human
SRC_MANUAL = "manual"
SRC_SAMPLE = "sample"

EVENT_COLS = ["fight_id", "fighter", "time_s", "category", "technique",
              "landed", "counter", "range", "source"]
FIGHT_COLS = ["fight_id", "name", "date", "duration_s", "fighter_a", "fighter_b",
              "pressure_a", "pressure_b", "notes", "source"]
COMBO_COLS = ["date", "training_type", "combination", "attempts", "successes",
              "notes", "source"]

# ========== storage ==========
import os
import tempfile

import pandas as pd



def _read(name, cols):
    p = DATA_DIR / f"{name}.csv"
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df:
            df[c] = ""
    return df[cols].copy()


def _write(name, df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, DATA_DIR / f"{name}.csv")  # atomic: no half-written files


# ---------- fights ----------
def load_fights():
    df = _read("fights", FIGHT_COLS)
    for c in ("duration_s", "pressure_a", "pressure_b"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_fights(df):
    _write("fights", df[FIGHT_COLS])


# ---------- events ----------
def load_events():
    df = _read("events", EVENT_COLS)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce").fillna(0.0)
    df["counter"] = df["counter"].astype(str).str.lower().eq("true")
    df["landed"] = df["landed"].where(df["landed"].isin(LANDED), "unknown")
    df["range"] = df["range"].where(df["range"].isin(RANGES), "unknown")
    df["category"] = df["category"].replace("", "unclassified")
    return df


def save_events(df):
    _write("events", df[EVENT_COLS])


def replace_events_for_fight(fight_id, new_events):
    ev = load_events()
    ev = ev[ev["fight_id"] != fight_id]
    save_events(pd.concat([ev, new_events[EVENT_COLS]], ignore_index=True))


# ---------- combos ----------
def load_combos():
    df = _read("combos", COMBO_COLS)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["successes"] = pd.to_numeric(df["successes"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def save_combos(df):
    out = df[COMBO_COLS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    _write("combos", out)


# ---------- housekeeping ----------
def remove_sample_data():
    fights, events, combos = load_fights(), load_events(), load_combos()
    gone = set(fights.loc[fights["source"] == "sample", "fight_id"])
    save_fights(fights[fights["source"] != "sample"])
    save_events(events[~events["fight_id"].isin(gone) & (events["source"] != "sample")])
    save_combos(combos[combos["source"] != "sample"])


def is_empty():
    return load_fights().empty and load_events().empty and load_combos().empty

# ========== stats ==========
import numpy as np
import pandas as pd


MIN_KNOWN_FOR_ACCURACY = 10  # don't show accuracy from a handful of labelled strikes

TABLE_COLS = ["Fight", "Fighter", "Punches", "Kicks", "Knees", "Elbows", "Total strikes",
              "Strikes/min", "Clinch actions", "Defensive actions", "Counters",
              "Accuracy %", "Strikes with landed info", "Unlabelled candidates",
              "Auto-detected %"]


def fight_minutes(fights, fight_id, events=None):
    row = fights[fights["fight_id"] == fight_id]
    d = float(row["duration_s"].iloc[0]) if len(row) and pd.notna(row["duration_s"].iloc[0]) else 0.0
    if d <= 0 and events is not None and len(events):
        d = float(events["time_s"].max())
    return d / 60.0


def fighter_fight_table(events, fights):
    if events.empty:
        return pd.DataFrame(columns=TABLE_COLS)
    names = fights.set_index("fight_id")["name"].to_dict()
    rows = []
    for (fid, fighter), g in events.groupby(["fight_id", "fighter"]):
        mins = fight_minutes(fights, fid, g)
        cat = g["category"].value_counts()
        strikes = g[g["category"].isin(STRIKE_CATS)]
        known = strikes[strikes["landed"].isin(["yes", "no"])]
        acc = (known["landed"] == "yes").mean() * 100 if len(known) >= MIN_KNOWN_FOR_ACCURACY else np.nan
        rows.append({
            "Fight": names.get(fid, fid), "Fighter": fighter,
            "Punches": int(cat.get("punch", 0)), "Kicks": int(cat.get("kick", 0)),
            "Knees": int(cat.get("knee", 0)), "Elbows": int(cat.get("elbow", 0)),
            "Total strikes": len(strikes),
            "Strikes/min": round(len(strikes) / mins, 2) if mins > 0 else np.nan,
            "Clinch actions": int(cat.get("clinch", 0)),
            "Defensive actions": int(cat.get("defence", 0)),
            "Counters": int(strikes["counter"].sum()),
            "Accuracy %": round(acc, 1) if pd.notna(acc) else np.nan,
            "Strikes with landed info": len(known),
            "Unlabelled candidates": int(cat.get("unclassified", 0)),
            "Auto-detected %": round(g["source"].isin([SRC_AUTO, SRC_AUTO_LABELLED]).mean() * 100, 1),
        })
    return pd.DataFrame(rows)[TABLE_COLS]


def strikes_per_minute_series(events):
    s = events[events["category"].isin(STRIKE_CATS)].copy()
    if s.empty:
        return pd.DataFrame(columns=["minute", "fighter", "strikes"])
    s["minute"] = (s["time_s"] // 60).astype(int) + 1
    return s.groupby(["minute", "fighter"]).size().reset_index(name="strikes")

# ========== video ==========
import cv2
import numpy as np
import pandas as pd


def _peaks(signal, thr, min_gap):
    """Local maxima above thr, at least min_gap samples apart (keeps the larger one)."""
    idx = [i for i in range(1, len(signal) - 1)
           if signal[i] > thr and signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]]
    kept = []
    for i in idx:
        if kept and i - kept[-1] < min_gap:
            if signal[i] > signal[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    return kept


def analyse_video(path, target_fps=8.0, width=320, sensitivity=1.5, min_gap_s=0.5, progress=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video file (unsupported codec?).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / target_fps)))
    times, left, right = [], [], []
    prev, idx = None, 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (width, max(1, int(h * width / w))))
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev is not None:
                mask = cv2.absdiff(gray, prev) > 25
                mid = mask.shape[1] // 2
                times.append(idx / fps)
                left.append(float(mask[:, :mid].mean()))
                right.append(float(mask[:, mid:].mean()))
            prev = gray
            if progress and total:
                progress(min(idx / total, 1.0))
        idx += 1
    cap.release()
    duration = idx / fps
    if len(times) < 10:
        raise ValueError("Video too short or unreadable for analysis.")

    sample_dt = step / fps
    timeline = pd.DataFrame({"time_s": times, "left": left, "right": right})
    win = max(3, int(3.0 / sample_dt))
    cands = []
    for side in ("left", "right"):
        s = timeline[side].rolling(3, center=True, min_periods=1).mean()
        s = s - s.rolling(win, center=True, min_periods=1).median()  # remove slow background/camera drift
        sig = s.to_numpy()
        thr = sig.mean() + sensitivity * sig.std()
        for i in _peaks(sig, thr, max(1, int(min_gap_s / sample_dt))):
            cands.append({"time_s": round(times[i], 1), "side": side, "strength": round(float(sig[i]), 4)})
    cands = pd.DataFrame(cands, columns=["time_s", "side", "strength"]).sort_values("time_s").reset_index(drop=True)

    warnings = []
    corr = np.corrcoef(left, right)[0, 1] if np.std(left) > 0 and np.std(right) > 0 else 0
    if corr > 0.8:
        warnings.append("Left and right motion are highly correlated - the camera is probably moving "
                        "or zooming. Candidate timings are unreliable for this video.")
    warnings.append("Fighters are assigned by side of the frame. This breaks down in the clinch, "
                    "when fighters swap sides, or if the camera pans.")
    return {"timeline": timeline, "candidates": cands, "duration_s": duration,
            "fps": fps, "warnings": warnings}

# ========== style ==========
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MIN_STRIKES = 30      # below this the result is "Inconclusive"
MIN_RANGE_EVENTS = 10

# value / cap, clipped to [0,1]
CAPS = {"punch_share": 0.55, "kick_share": 0.50, "knee_share": 0.25, "clinch_rate": 0.15,
        "counter_share": 0.35, "defence_share": 0.35, "offence_rate": 25.0,
        "close_share": 0.60, "long_share": 0.50, "pressure": 100.0, "accuracy": 1.0}

# style -> list of (feature, weight, inverted)
STYLES = {
    "Muay Mat (pressure puncher)": [("punch_share", .40, False), ("pressure", .25, False),
                                    ("offence_rate", .20, False), ("defence_share", .15, True)],
    "Muay Khao (knee/clinch)":     [("knee_share", .35, False), ("clinch_rate", .30, False),
                                    ("close_share", .20, False), ("pressure", .15, False)],
    "Muay Tae (kicker)":           [("kick_share", .50, False), ("long_share", .25, False),
                                    ("pressure", .25, True)],
    "Muay Femur (technical counter)": [("counter_share", .35, False), ("defence_share", .30, False),
                                       ("pressure", .10, True), ("accuracy", .15, False),
                                       ("offence_rate", .10, True)],
}

LABELS = {
    "punch_share": ("Punches as share of strikes", "pct"), "kick_share": ("Kicks as share of strikes", "pct"),
    "knee_share": ("Knees as share of strikes", "pct"), "clinch_rate": ("Clinch actions as share of all actions", "pct"),
    "counter_share": ("Strikes thrown as counters", "pct"), "defence_share": ("Defensive actions vs strikes+defence", "pct"),
    "offence_rate": ("Strikes per minute", "num"), "close_share": ("Actions at close range", "pct"),
    "long_share": ("Actions at long range", "pct"), "pressure": ("Forward pressure (time advancing)", "pctraw"),
    "accuracy": ("Strike accuracy", "pct"),
}


def _fmt(key, v):
    kind = LABELS[key][1]
    return f"{v:.0f}%" if kind == "pctraw" else f"{v*100:.0f}%" if kind == "pct" else f"{v:.1f}"


@dataclass
class StyleResult:
    verdict: str
    percentages: dict = field(default_factory=dict)
    confidence: str = "n/a"
    confidence_score: float = 0.0
    n_strikes: int = 0
    features: dict = field(default_factory=dict)
    explanation: list = field(default_factory=list)
    limitations: list = field(default_factory=list)


def extract_features(events, fights, fighter):
    ev = events[(events["fighter"] == fighter) & (events["category"] != "unclassified")]
    strikes = ev[ev["category"].isin(STRIKE_CATS)]
    n = len(strikes)
    f = {k: np.nan for k in CAPS}
    if n == 0:
        return f, ev
    f["punch_share"] = (strikes["category"] == "punch").mean()
    f["kick_share"] = (strikes["category"] == "kick").mean()
    f["knee_share"] = (strikes["category"] == "knee").mean()
    f["clinch_rate"] = (ev["category"] == "clinch").mean()
    f["counter_share"] = strikes["counter"].mean()
    d = (ev["category"] == "defence").sum()
    f["defence_share"] = d / (n + d)
    rng = ev[ev["category"].isin(STRIKE_CATS + ["clinch"]) & (ev["range"] != "unknown")]
    if len(rng) >= MIN_RANGE_EVENTS:
        f["close_share"] = (rng["range"] == "close").mean()
        f["long_share"] = (rng["range"] == "long").mean()
    known = strikes[strikes["landed"].isin(["yes", "no"])]
    if len(known) >= MIN_RANGE_EVENTS:
        f["accuracy"] = (known["landed"] == "yes").mean()
    # minutes + pressure from fight metadata (pressure is manual input)
    mins, pw, pt = 0.0, 0.0, 0.0
    for fid in ev["fight_id"].unique():
        row = fights[fights["fight_id"] == fid]
        if row.empty:
            continue
        r = row.iloc[0]
        dur = r["duration_s"] if pd.notna(r["duration_s"]) and r["duration_s"] > 0 else ev.loc[ev["fight_id"] == fid, "time_s"].max()
        mins += dur / 60
        p = r["pressure_a"] if r["fighter_a"] == fighter else r["pressure_b"] if r["fighter_b"] == fighter else np.nan
        if pd.notna(p):
            pw += p * dur
            pt += dur
    f["offence_rate"] = n / mins if mins > 0 else np.nan
    f["pressure"] = pw / pt if pt > 0 else np.nan
    return f, ev


def _norm(key, v, invert):
    x = min(max(v / CAPS[key], 0.0), 1.0)
    return 1 - x if invert else x


def classify(events, fights, fighter):
    f, ev = extract_features(events, fights, fighter)
    n = int((ev["category"].isin(STRIKE_CATS)).sum())
    unl = int(((events["fighter"] == fighter) & (events["category"] == "unclassified")).sum())
    lim = ["Rule-based heuristic, not a trained model: it describes what the recorded numbers look like, "
           "not what the fighter 'is'.",
           "Style depends on opponent, rules, weight class and game plan; a few fights can mislead."]
    if unl:
        lim.append(f"{unl} unlabelled auto-detected candidate(s) were ignored.")
    if n < MIN_STRIKES:
        return StyleResult("Inconclusive", n_strikes=n, features=f, confidence="n/a",
                           explanation=[f"Only {n} labelled strikes; at least {MIN_STRIKES} are needed."],
                           limitations=lim)

    scores, contrib, missing = {}, {}, set()
    for style, parts in STYLES.items():
        tot_w = sum(w for _, w, _ in parts)
        used = [(k, w, inv) for k, w, inv in parts if pd.notna(f[k])]
        missing |= {k for k, _, _ in parts if pd.isna(f[k])}
        if sum(w for _, w, _ in used) < 0.5 * tot_w:
            scores[style] = 0.0
            continue
        uw = sum(w for _, w, _ in used)
        contrib[style] = sorted(((k, w / uw * _norm(k, f[k], inv), inv) for k, w, inv in used),
                                key=lambda t: -t[1])
        scores[style] = sum(c for _, c, _ in contrib[style])
    total = sum(scores.values())
    pct = {s: (v / total if total > 0 else 1 / len(scores)) for s, v in scores.items()}
    ranked = sorted(pct.items(), key=lambda t: -t[1])
    (s1, p1), (s2, p2) = ranked[0], ranked[1]
    margin = p1 - p2
    mixed = p1 < 0.35 or margin < 0.08
    verdict = "Mixed style" if mixed else s1

    conf_score = 0.5 * min(n / 150, 1) + 0.5 * min(margin / 0.20, 1)
    if mixed:
        conf_score = min(conf_score, 0.6)
    conf = "high" if conf_score >= 0.7 else "medium" if conf_score >= 0.4 else "low"

    def evidence(style):
        out = []
        for k, c, inv in contrib.get(style, [])[:3]:
            lab = LABELS[k][0]
            out.append(f"{lab}: {_fmt(k, f[k])}" + (" (low value supports this style)" if inv else ""))
        return out

    if mixed:
        exp = [f"No style clearly dominates: {s1.split(' (')[0]} {p1*100:.0f}% vs {s2.split(' (')[0]} {p2*100:.0f}%."]
        exp += ["Closest styles - " + s1.split(" (")[0] + ": " + "; ".join(evidence(s1))]
        exp += [s2.split(" (")[0] + ": " + "; ".join(evidence(s2))]
    else:
        exp = [f"{s1.split(' (')[0]} leads with {p1*100:.0f}% (next: {s2.split(' (')[0]} {p2*100:.0f}%)."]
        exp += ["Key evidence: " + "; ".join(evidence(s1))]
    if missing:
        lim.append("Missing inputs (not scored): " + ", ".join(LABELS[m][0] for m in sorted(missing))
                   + ". Record range / forward pressure / landed info for a better estimate.")
    lim.append(f"Confidence is based on sample size ({n} strikes) and the gap between the top two styles.")
    return StyleResult(verdict, pct, conf, conf_score, n, f, exp, lim)

# ========== combos ==========
import re

import numpy as np
import pandas as pd

KEYWORDS = {"punch": ["jab", "cross", "hook", "uppercut", "overhand", "punch", "hammerfist"],
            "kick": ["kick", "teep"], "knee": ["knee"], "elbow": ["elbow"],
            "defence": ["check", "block", "parry", "slip", "evade", "catch", "duck", "slide"]}


def normalize(text):
    parts = [re.sub(r"\s+", " ", p.strip().lower()) for p in re.split(r"[-–>,/]+", str(text))]
    return "-".join(p for p in parts if p)


def categorize(token):
    for cat, words in KEYWORDS.items():
        if any(w in token for w in words):
            return cat
    return "unknown"


def categories(combo):
    return [categorize(t) for t in combo.split("-")]


def wilson_lower(s, n, z=1.96):
    """Conservative success estimate: 3/3 should not outrank 80/100."""
    if n == 0:
        return 0.0
    p = s / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / d


def prepare(df, training_types=None):
    d = df.dropna(subset=["date"]).copy()
    d["combination"] = d["combination"].map(normalize)
    if training_types:
        d = d[d["training_type"].isin(training_types)]
    return d


def summary(df, min_attempts=0):
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("combination").agg(attempts=("attempts", "sum"), successes=("successes", "sum"),
                                      sessions=("date", "nunique"), last_trained=("date", "max")).reset_index()
    g = g[g["attempts"] >= min_attempts]
    g["success_rate_%"] = (g["successes"] / g["attempts"].replace(0, np.nan) * 100).round(1)
    g["reliable_rate_%"] = [round(wilson_lower(s, n) * 100, 1) for s, n in zip(g["successes"], g["attempts"])]
    g["structure"] = g["combination"].map(lambda c: " > ".join(categories(c)))
    g["trend_pts"] = g["combination"].map(lambda c: improvement(df, c))
    g["last_trained"] = g["last_trained"].dt.strftime("%Y-%m-%d")
    return g.sort_values("reliable_rate_%", ascending=False).reset_index(drop=True)


def per_date(df, combos=None):
    d = df if combos is None else df[df["combination"].isin(combos)]
    g = d.groupby(["date", "combination"]).agg(attempts=("attempts", "sum"), successes=("successes", "sum")).reset_index()
    g["rate_%"] = g["successes"] / g["attempts"].replace(0, np.nan) * 100
    return g


def improvement(df, combo):
    """Late-half success rate minus early-half rate, in percentage points (needs >=4 sessions)."""
    g = per_date(df, [combo]).sort_values("date")
    if len(g) < 4:
        return np.nan
    k = len(g) // 2
    early, late = g.iloc[:k], g.iloc[-k:]
    if early["attempts"].sum() == 0 or late["attempts"].sum() == 0:
        return np.nan
    return round((late["successes"].sum() / late["attempts"].sum()
                  - early["successes"].sum() / early["attempts"].sum()) * 100, 1)

# ========== sample_data ==========
from datetime import date, timedelta

import numpy as np
import pandas as pd


PROFILES = {
    "Sample Khao":   dict(n=120, cat=[.25, .25, .45, .05], clinch=40, defence=15, counter=.08, landed=.55, rng=[.1, .3, .6]),
    "Sample Femur":  dict(n=85,  cat=[.40, .30, .12, .18], clinch=8,  defence=45, counter=.42, landed=.62, rng=[.3, .6, .1]),
    "Sample Mat":    dict(n=210, cat=[.70, .15, .10, .05], clinch=10, defence=12, counter=.10, landed=.50, rng=[.1, .5, .4]),
    "Sample Tae":    dict(n=110, cat=[.15, .70, .08, .07], clinch=6,  defence=20, counter=.15, landed=.50, rng=[.55, .4, .05]),
}
FIGHTS = [("smp00001", "Sample: Khao vs Femur", "Sample Khao", "Sample Femur", 78, 42),
          ("smp00002", "Sample: Mat vs Tae", "Sample Mat", "Sample Tae", 85, 40)]


def _events(rng, fid, fighter, dur, p):
    rows = []
    rngs = ["long", "mid", "close"]
    for _ in range(p["n"]):
        cat = rng.choice(STRIKE_CATS, p=p["cat"])
        rows.append((fid, fighter, round(rng.uniform(0, dur), 1), cat, rng.choice(TECHNIQUES[cat]),
                     "yes" if rng.random() < p["landed"] else "no", bool(rng.random() < p["counter"]),
                     rng.choice(rngs, p=p["rng"]), SRC_SAMPLE))
    for cat, k in (("clinch", p["clinch"]), ("defence", p["defence"])):
        for _ in range(k):
            rows.append((fid, fighter, round(rng.uniform(0, dur), 1), cat, rng.choice(TECHNIQUES[cat]),
                         "unknown", False, rng.choice(rngs, p=p["rng"] if cat == "clinch" else [.3, .5, .2]), SRC_SAMPLE))
    return rows


def build_fights_events():
    rng = np.random.default_rng(7)
    fights, events = [], []
    for fid, name, a, b, pa, pb in FIGHTS:
        fights.append((fid, name, str(date.today() - timedelta(days=14)), 900.0, a, b, pa, pb,
                       "Example data - not a real fight", SRC_SAMPLE))
        for fighter in (a, b):
            events += _events(rng, fid, fighter, 900.0, PROFILES[fighter])
    return (pd.DataFrame(fights, columns=FIGHT_COLS),
            pd.DataFrame(events, columns=EVENT_COLS).sort_values(["fight_id", "time_s"]))


def build_combos():
    rng = np.random.default_rng(11)
    combos = {"jab-cross-low kick": (.35, .70), "jab-cross-hook-kick": (.25, .60),
              "teep-cross-hook": (.40, .55), "jab-low kick": (.50, .72),
              "cross-knee-elbow": (.20, .35), "jab-cross-hook-low kick": (.30, .50)}
    n_sessions, rows = 10, []
    for s in range(n_sessions):
        d = date.today() - timedelta(days=7 * (n_sessions - 1 - s))
        ttype = "pad work" if s % 2 == 0 else "sparring"
        for name, (p0, p1) in combos.items():
            if rng.random() < 0.25:
                continue
            p = p0 + (p1 - p0) * s / (n_sessions - 1) - (0.12 if ttype == "sparring" else 0)
            att = int(rng.integers(20, 31) if ttype == "pad work" else rng.integers(6, 13))
            suc = int(rng.binomial(att, min(max(p, 0.02), 0.98)))
            rows.append((str(d), ttype, name, att, suc, "", SRC_SAMPLE))
    return pd.DataFrame(rows, columns=COMBO_COLS)


def load_samples():
    remove_sample_data()  # idempotent: never duplicate examples
    f, e = build_fights_events()
    c = build_combos()
    save_fights(pd.concat([load_fights(), f], ignore_index=True))
    save_events(pd.concat([load_events(), e], ignore_index=True))
    c["date"] = pd.to_datetime(c["date"])
    save_combos(pd.concat([load_combos(), c], ignore_index=True))


def seed_once():
    """First launch only: populate example data so the app is never an empty shell."""
    if not SEED_FLAG.exists() and is_empty():
        load_samples()
    SEED_FLAG.parent.mkdir(parents=True, exist_ok=True)
    SEED_FLAG.touch()

# ========== common ==========
import streamlit as st

CSS = """<style>
.block-container{padding:1rem .75rem 4rem;max-width:760px}
.stTabs [data-baseweb="tab-list"]{overflow-x:auto;gap:0}
.stTabs [data-baseweb="tab"]{padding:.5rem .7rem;font-size:.9rem}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{width:100%;min-height:2.8rem}
h1{font-size:1.5rem!important} h2,h3{font-size:1.15rem!important}
</style>"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def show(fig, height=320, **_):
    """Render a Plotly figure sized for a phone: short, legend below, no toolbar."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8), title_font_size=14,
                      legend=dict(orientation="h", y=-0.25, x=0, title=None))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ========== fight_tab ==========
import os
import tempfile
import uuid
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st


LABEL_OPTIONS = ["punch", "kick", "knee", "elbow", "clinch", "defence"]


def _mmss(t):
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def _create_fight_panel(fights):
    with st.expander("➕ Add a new fight", expanded=fights.empty):
        name = st.text_input("Fight name", placeholder="e.g. Gym sparring, round 3")
        fdate = st.date_input("Date", value=date.today())
        fa = st.text_input("Fighter A (left side of frame)", "Fighter A")
        fb = st.text_input("Fighter B (right side of frame)", "Fighter B")
        dur_min = st.number_input("Duration (minutes)", 0.5, 300.0, 3.0, step=0.5,
                                  help="Used for strikes/min. Overridden by the video length if you upload one.")
        up = st.file_uploader("Fight video (optional, MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])
        sens = 1.5
        if up:
            sens = st.slider("Detection sensitivity (lower = more candidates)", 0.5, 3.0, 1.5, 0.1)
            st.caption("Video analysis finds motion bursts as *candidate* actions (timing only). "
                       "You label them afterwards - techniques are not detected automatically.")
        if st.button("Create fight", type="primary"):
            if not name.strip() or not fa.strip() or not fb.strip() or fa.strip() == fb.strip():
                st.error("Enter a fight name and two different fighter names.")
                return
            fid, duration, events = uuid.uuid4().hex[:8], dur_min * 60, pd.DataFrame(columns=EVENT_COLS)
            if up:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
                    tmp.write(up.getbuffer())
                bar = st.progress(0.0, text="Analysing video...")
                try:
                    res = analyse_video(tmp.name, sensitivity=sens,
                                              progress=lambda p: bar.progress(p, text="Analysing video..."))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Video analysis failed: {e}")
                    return
                finally:
                    os.unlink(tmp.name)
                bar.empty()
                duration, c = res["duration_s"], res["candidates"]
                events = pd.DataFrame({"fight_id": fid, "time_s": c["time_s"],
                                       "fighter": c["side"].map({"left": fa.strip(), "right": fb.strip()}),
                                       "category": "unclassified", "technique": "", "landed": "unknown",
                                       "counter": False, "range": "unknown", "source": SRC_AUTO})
                st.session_state["warnings"] = res["warnings"]
                st.session_state["timeline"] = (fid, res["timeline"])
            save_fights(pd.concat([fights, pd.DataFrame([{
                "fight_id": fid, "name": name.strip(), "date": str(fdate), "duration_s": duration,
                "fighter_a": fa.strip(), "fighter_b": fb.strip(), "pressure_a": None, "pressure_b": None,
                "notes": "", "source": SRC_MANUAL}])], ignore_index=True))
            save_events(pd.concat([load_events(), events[EVENT_COLS]], ignore_index=True))
            st.session_state["selected_fight"] = fid
            st.rerun()


def _label_queue(fid, events):
    """One-tap labelling of auto-detected candidates, one at a time (phone friendly)."""
    unl = events[(events["fight_id"] == fid) & (events["category"] == "unclassified")].sort_values("time_s")
    if unl.empty:
        return
    row = unl.iloc[0]
    st.markdown(f"### Label candidates ({len(unl)} left)")
    st.info(f"**{row['fighter']}** at **{_mmss(row['time_s'])}** - what was it?")
    pick = st.pills("Action", [c.title() for c in LABEL_OPTIONS], key=f"lab_{row.name}", label_visibility="collapsed")
    if pick:
        ev = events.copy()
        ev.loc[row.name, ["category", "source"]] = [pick.lower(), SRC_AUTO_LABELLED]
        save_events(ev)
        st.rerun()
    if st.button("🗑 Not an action - delete", key=f"del_{row.name}"):
        save_events(events.drop(row.name))
        st.rerun()


def _quick_add(fight, fid):
    fighters = [fight["fighter_a"], fight["fighter_b"]]
    with st.expander("➕ Quick-add an event"):
        with st.form(f"qa_{fid}", clear_on_submit=True):
            who = st.radio("Fighter", fighters, horizontal=True)
            t = st.number_input("Time (seconds into fight)", 0.0, 100000.0, 0.0, step=1.0)
            cat = st.selectbox("Category", [c for c in ALL_CATS if c != "unclassified"])
            tech = st.selectbox("Technique (optional)", [""] + ALL_TECHS)
            landed = st.radio("Landed?", LANDED, horizontal=True)
            rng = st.radio("Range", RANGES, horizontal=True)
            counter = st.checkbox("Counter-attack")
            if st.form_submit_button("Add event", type="primary"):
                new = pd.DataFrame([{"fight_id": fid, "fighter": who, "time_s": t, "category": cat,
                                     "technique": tech, "landed": landed, "counter": counter,
                                     "range": rng, "source": SRC_MANUAL}])
                save_events(pd.concat([load_events(), new], ignore_index=True))
                st.rerun()


def _table_editor(fight, events):
    fid = fight["fight_id"]
    fighters = [fight["fighter_a"], fight["fighter_b"]]
    ev = events[events["fight_id"] == fid].sort_values("time_s").reset_index(drop=True)
    with st.expander("✏ Edit all events (table - easier on a larger screen)"):
        edited = st.data_editor(
            ev.drop(columns=["fight_id"]), num_rows="dynamic", hide_index=True, use_container_width=True,
            key=f"ed_{fid}", disabled=["source"],
            column_config={
                "fighter": st.column_config.SelectboxColumn(options=fighters, required=True),
                "time_s": st.column_config.NumberColumn("time (s)", min_value=0.0, step=0.5, format="%.1f"),
                "category": st.column_config.SelectboxColumn(options=ALL_CATS, required=True),
                "technique": st.column_config.SelectboxColumn(options=[""] + ALL_TECHS),
                "landed": st.column_config.SelectboxColumn(options=LANDED),
                "counter": st.column_config.CheckboxColumn("counter?"),
                "range": st.column_config.SelectboxColumn(options=RANGES),
            })
        if st.button("💾 Save table changes", key=f"save_{fid}"):
            out = edited[edited["fighter"].isin(fighters)].copy()
            out["fight_id"] = fid
            out["time_s"] = out["time_s"].fillna(0.0)
            out["category"] = out["category"].fillna("unclassified")
            out["technique"] = out["technique"].fillna("")
            out["landed"] = out["landed"].fillna("unknown")
            out["range"] = out["range"].fillna("unknown")
            out["counter"] = out["counter"].fillna(False).astype(bool)
            out["source"] = out["source"].fillna(SRC_MANUAL)
            relabel = (out["source"] == SRC_AUTO) & (out["category"] != "unclassified")
            out.loc[relabel, "source"] = SRC_AUTO_LABELLED
            replace_events_for_fight(fid, out)
            st.rerun()


def _fight_settings(fight, fights):
    key = fight["fight_id"]
    pa = st.number_input(f"{fight['fighter_a']} - forward pressure (% of time advancing)", 0.0, 100.0,
                         None if pd.isna(fight["pressure_a"]) else float(fight["pressure_a"]), step=5.0, key=f"pa_{key}")
    pb = st.number_input(f"{fight['fighter_b']} - forward pressure (% of time advancing)", 0.0, 100.0,
                         None if pd.isna(fight["pressure_b"]) else float(fight["pressure_b"]), step=5.0, key=f"pb_{key}")
    cur = float(fight["duration_s"]) / 60 if pd.notna(fight["duration_s"]) and fight["duration_s"] > 0 else 3.0
    dur = st.number_input("Fight duration (minutes)", 0.5, 300.0, cur, step=0.5, key=f"dur_{key}")
    if st.button("Save fight settings", key=f"fs_{key}"):
        fights.loc[fights["fight_id"] == key, ["pressure_a", "pressure_b", "duration_s"]] = [pa, pb, dur * 60]
        save_fights(fights)
        st.rerun()
    st.caption("Forward pressure is a manual estimate (not detected from video) and feeds the style classifier.")


def render_fight_tab():
    fights, events = load_fights(), load_events()
    _create_fight_panel(fights)
    if fights.empty:
        st.info("No fights yet. Add one above or load example data at the bottom of the page.")
        return

    labels = {r.fight_id: f"{r.name} ({r.date})" for r in fights.itertuples()}
    ids = list(labels)
    default = st.session_state.get("selected_fight")
    sel = st.selectbox("Fight", ids, index=ids.index(default) if default in ids else 0, format_func=labels.get)
    st.session_state["selected_fight"] = sel
    fight = fights[fights["fight_id"] == sel].iloc[0]
    fev = events[events["fight_id"] == sel]

    n_auto = int(fev["source"].isin([SRC_AUTO, SRC_AUTO_LABELLED]).sum())
    n_unl = int((fev["category"] == "unclassified").sum())
    st.caption(f"Fight source: **{fight['source']}** · {len(fev)} events · {n_auto} from video detection · "
               f"{n_unl} unlabelled (excluded from stats)")
    for w in st.session_state.get("warnings", []) if st.session_state.get("timeline", (None,))[0] == sel else []:
        st.warning(w)

    _label_queue(sel, events)
    tl = st.session_state.get("timeline")
    if tl and tl[0] == sel:
        show(px.line(tl[1].melt("time_s", var_name="side", value_name="motion"), x="time_s", y="motion",
                     color="side", title="Motion activity (auto-detected)"), 260)

    _quick_add(fight, sel)
    _table_editor(fight, events)
    with st.expander("⚙ Fight settings (duration, forward pressure)"):
        _fight_settings(fight, fights)

    st.divider()
    table = fighter_fight_table(fev, fights)
    if table.empty:
        st.info("No events recorded for this fight yet.")
        return
    st.markdown("### Statistics")
    st.dataframe(table[["Fighter", "Total strikes", "Strikes/min", "Accuracy %", "Counters"]],
                 hide_index=True, use_container_width=True)
    with st.expander("Full table"):
        st.dataframe(table.drop(columns=["Fight"]), hide_index=True, use_container_width=True)
        st.caption("Accuracy is shown only with ≥10 strikes that have landed yes/no filled in; "
                   "unlabelled auto candidates are excluded.")
    long = table.melt(id_vars="Fighter", value_vars=["Punches", "Kicks", "Knees", "Elbows"],
                      var_name="Technique", value_name="Count")
    show(px.bar(long, x="Technique", y="Count", color="Fighter", barmode="group", title="Strikes by type"))
    spm = strikes_per_minute_series(fev)
    if not spm.empty:
        show(px.line(spm, x="minute", y="strikes", color="fighter", markers=True, title="Strikes per minute"))
    oc = table.melt(id_vars="Fighter", value_vars=["Total strikes", "Defensive actions", "Clinch actions"],
                    var_name="Action", value_name="Count")
    show(px.bar(oc, x="Fighter", y="Count", color="Action", barmode="group", title="Offence vs defence vs clinch"))

    st.download_button("⬇ Stats CSV (all fights)", fighter_fight_table(events, fights).to_csv(index=False),
                       "fight_statistics.csv", "text/csv")
    st.download_button("⬇ Event log CSV (all fights)", events.to_csv(index=False), "fight_events.csv", "text/csv")

# ========== style_tab ==========
import pandas as pd
import plotly.express as px
import streamlit as st




def render_style_tab():
    fights, events = load_fights(), load_events()
    st.subheader("Fighting Style Classifier")
    st.info("This is a transparent rule-based estimate from recorded statistics, not a verdict. "
            "Small samples, one opponent or one game plan can easily mislead. "
            "Too little data returns **Inconclusive** by design.")
    if events.empty:
        st.info("No events yet. Add fights in the first tab or load the example data.")
        return

    fighters = sorted(events["fighter"].unique())
    fighter = st.selectbox("Fighter", fighters)
    names = fights.set_index("fight_id")["name"].to_dict()
    fids = sorted(events.loc[events["fighter"] == fighter, "fight_id"].unique())
    chosen = st.multiselect("Fights to include", fids, default=fids, format_func=lambda x: names.get(x, x))
    ev = events[events["fight_id"].isin(chosen)]
    if ev.empty:
        st.warning("Select at least one fight.")
        return

    res = classify(ev, fights, fighter)
    if res.verdict == "Inconclusive":
        st.warning("### Inconclusive")
    else:
        st.success(f"### {res.verdict}")
        st.write(f"Confidence: **{res.confidence}** ({res.confidence_score*100:.0f}/100) · "
                 f"based on {res.n_strikes} labelled strikes")

    if res.percentages:
        d = pd.DataFrame({"Style": [s.split(" (")[0] for s in res.percentages],
                          "Share %": [round(v * 100, 1) for v in res.percentages.values()]}).sort_values("Share %")
        show(px.bar(d, x="Share %", y="Style", orientation="h", text="Share %",
                               title="Style distribution (shares of the score, sum = 100%)"),
                        use_container_width=True)
    st.markdown("**Why**")
    for line in res.explanation:
        st.write("- " + line)
    with st.expander("Measured features"):
        rows = [{"Feature": LABELS[k][0], "Value": "not available" if pd.isna(v) else _fmt(k, v)}
                for k, v in res.features.items()]
        ft = pd.DataFrame(rows)
        st.dataframe(ft, hide_index=True, use_container_width=True)
        st.download_button("⬇ Features CSV", ft.to_csv(index=False), f"{fighter}_style_features.csv", "text/csv")
    with st.expander("Limitations", expanded=True):
        for line in res.limitations:
            st.write("- " + line)

    st.markdown("### All fighters")
    rows = []
    for f in fighters:
        r = classify(events, fights, f)
        top = max(r.percentages.items(), key=lambda t: t[1]) if r.percentages else ("-", 0)
        rows.append({"Fighter": f, "Verdict": r.verdict, "Confidence": r.confidence, "Strikes": r.n_strikes,
                     "Top-scoring style": top[0].split(" (")[0], "Top share %": round(top[1] * 100, 1)})
    out = pd.DataFrame(rows)
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.download_button("⬇ Classification CSV", out.to_csv(index=False), "style_classification.csv", "text/csv")

# ========== combo_tab ==========
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st




def _add_form(df):
    with st.form("add_combo"):
        c1, c2 = st.columns([2, 1])
        combo = c1.text_input("Combination", placeholder="jab-cross-low kick")
        ttype = c2.selectbox("Training type", TRAINING_TYPES)
        c3, c4, c5 = st.columns(3)
        d = c3.date_input("Date", date.today())
        att = c4.number_input("Attempts", 1, 10000, 10)
        suc = c5.number_input("Successful", 0, 10000, 5)
        notes = st.text_input("Notes (optional)")
        ok = st.form_submit_button("Save entry", type="primary")
    if ok:
        norm = normalize(combo)
        if not norm:
            st.error("Enter a combination, e.g. jab-cross-low kick.")
        elif suc > att:
            st.error("Successful executions cannot exceed attempts.")
        else:
            row = pd.DataFrame([{"date": pd.Timestamp(d), "training_type": ttype, "combination": norm,
                                 "attempts": int(att), "successes": int(suc), "notes": notes, "source": SRC_MANUAL}])
            save_combos(pd.concat([df, row], ignore_index=True))
            cats = categories(norm)
            st.success(f"Saved: {norm}  ({' > '.join(cats)})")
            if "unknown" in cats:
                st.warning("Some techniques were not recognised and are categorised as 'unknown'.")
            st.rerun()


def render_combo_tab():
    df = load_combos()
    st.subheader("Combination Effectiveness")
    st.caption("All data here is entered manually (or flagged `sample`).")
    _add_form(df)
    if df.empty:
        st.info("No entries yet.")
        return

    c1, c2 = st.columns([2, 1])
    types = c1.multiselect("Training type", TRAINING_TYPES, default=TRAINING_TYPES)
    min_att = c2.slider("Min. total attempts", 0, 100, 10)
    data = prepare(df, types)
    summ = summary(data, min_att)
    if summ.empty:
        st.warning("No combinations match these filters.")
        return

    best = summ.iloc[0]
    st.success(f"Most effective: **{best['combination']}** - {best['success_rate_%']}% "
               f"({best['successes']}/{best['attempts']}); reliable estimate {best['reliable_rate_%']}%")
    st.caption("Ranking uses a conservative estimate (Wilson lower bound) so 3/3 does not outrank 80/100. "
               "trend_pts = later-half rate minus earlier-half rate (needs ≥4 sessions).")
    st.dataframe(summ[["combination", "attempts", "successes", "success_rate_%", "reliable_rate_%", "trend_pts"]],
                 hide_index=True, use_container_width=True)
    with st.expander("All columns"):
        st.dataframe(summ, hide_index=True, use_container_width=True)
    st.download_button("⬇ Summary CSV", summ.to_csv(index=False), "combo_summary.csv", "text/csv")

    st.markdown("### Compare")
    pick = st.multiselect("Combinations", list(summ["combination"]), default=list(summ["combination"][:3]))
    if pick:
        cmp = data[data["combination"].isin(pick)].groupby(["combination", "training_type"]).agg(
            attempts=("attempts", "sum"), successes=("successes", "sum")).reset_index()
        cmp["success_rate_%"] = (cmp["successes"] / cmp["attempts"] * 100).round(1)
        show(px.bar(cmp, x="combination", y="success_rate_%", color="training_type",
                               barmode="group", title="Success rate by training type"), use_container_width=True)
        trend = per_date(data, pick)
        show(px.line(trend, x="date", y="rate_%", color="combination", markers=True,
                                title="Success rate over time"), use_container_width=True)
        st.caption("Sparring sessions usually have fewer attempts and lower rates than pad work - "
                   "filter by training type to compare like with like.")

    all_trend = data.groupby("date").agg(a=("attempts", "sum"), s=("successes", "sum")).reset_index()
    all_trend["overall_rate_%"] = all_trend["s"] / all_trend["a"] * 100
    show(px.line(all_trend, x="date", y="overall_rate_%", markers=True,
                            title="Overall success rate per session date (all combinations)"),
                    use_container_width=True)

    with st.expander("Edit / delete entries"):
        ed = df.copy()
        ed["date"] = ed["date"].dt.date
        edited = st.data_editor(ed, num_rows="dynamic", hide_index=True, use_container_width=True,
                                key="combo_editor", disabled=["source"],
                                column_config={"training_type": st.column_config.SelectboxColumn(options=TRAINING_TYPES),
                                               "date": st.column_config.DateColumn()})
        if st.button("💾 Save log changes"):
            out = edited.dropna(subset=["combination", "date"]).copy()
            out["combination"] = out["combination"].map(normalize)
            out["attempts"] = out["attempts"].fillna(0).astype(int)
            out["successes"] = out["successes"].fillna(0).astype(int)
            out["training_type"] = out["training_type"].fillna(TRAINING_TYPES[0])
            out["notes"] = out["notes"].fillna("")
            out["source"] = out["source"].fillna(SRC_MANUAL)
            if (out["successes"] > out["attempts"]).any():
                st.error("Some rows have more successes than attempts.")
            else:
                out["date"] = pd.to_datetime(out["date"])
                save_combos(out)
                st.rerun()
        st.download_button("⬇ Full log CSV", df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False),
                           "combo_log.csv", "text/csv")

# ========== app ==========
import streamlit as st


st.set_page_config(page_title="Fight Intelligence", page_icon="🥊", layout="centered",
                   initial_sidebar_state="collapsed")
inject_css()
seed_once()

st.title("🥊 Fight Intelligence")
t1, t2, t3 = st.tabs(["📊 Stats", "🧭 Style", "🥋 Combos"])
with t1:
    render_fight_tab()
with t2:
    render_style_tab()
with t3:
    render_combo_tab()

with st.expander("⚙ Data & example data"):
    st.caption(f"Stored in `{DATA_DIR}`. Records are labelled by source: auto-detected, "
               "auto+manual label, manual or sample.")
    if st.button("Load example data"):
        load_samples()
        st.rerun()
    if st.button("Remove example data"):
        remove_sample_data()
        st.rerun()
    st.caption("Example data is synthetic and tagged `sample`; removing it never touches your own data.")
