"""
Step 6: Live predictions with a public track record.

Run from the project folder:
    python src/live.py

Every run does two things:

  1. SCORE: looks for routine inspections that happened since the last run and
     compares each result to the risk score that was saved BEFORE that
     inspection took place. Results are added to live/outcomes.csv.
  2. PREDICT: gives every active restaurant a risk score as of today and saves
     it to live/snapshots/YYYY-MM-DD.csv.

The model is frozen (models/gradient_boosting.joblib) and never retrained here.
Snapshots are committed to git, so the commit date proves each prediction was
made before the inspection result existed. A scheduled GitHub Action
(.github/workflows/live.yml) runs this every week.

Outputs:
  live/snapshots/YYYY-MM-DD.csv   one risk score per restaurant, per run
  live/outcomes.csv               each new inspection matched to its earlier score
  reports/live_track_record.md    running results

Replaying the past (for testing only, never mixed into the live record):
    python src/live.py --raw data/raw/food_inspections.csv --as-of 2026-09-01 --out-dir replay
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from clean import clean_inspections, standardize_columns
from download_data import download_all
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "live"
MODEL_PATH = ROOT / "models" / "gradient_boosting.joblib"
REPORT_PATH = ROOT / "reports" / "live_track_record.md"

ACTIVE_DAYS = 3 * 365     # a restaurant counts as active if it was inspected in the last 3 years
MAX_LEAD_DAYS = 45        # ignore a prediction older than this when the inspection happens
STALE_DAYS = 30           # warn if the newest inspection in the data is older than this
MIN_SCORED, MIN_FAILS = 200, 30   # below this, the report says the numbers are still noisy
CLOSED_RESULTS = {"Out of Business", "Business Not Located"}

SNAPSHOT_COLS = ["license_number", "risk_score", "risk_percentile", "last_routine_failed"]
OUTCOME_COLS = ["inspection_id", "license_number", "dba_name", "inspection_date", "results", "failed",
                "snapshot_date", "lead_days", "risk_score", "risk_percentile", "last_routine_failed"]


# ---------- Getting the data ready ----------

def closed_licenses(raw: pd.DataFrame) -> set[int]:
    """Licenses whose most recent record says the business is gone."""
    df = standardize_columns(raw.copy())
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="mixed", errors="coerce")
    df["license_number"] = pd.to_numeric(df["license_number"], errors="coerce")
    df = df.dropna(subset=["inspection_date", "license_number"]).sort_values("inspection_date", kind="stable")
    last = df.groupby("license_number").tail(1)
    gone = last[last["results"].fillna("").str.strip().isin(CLOSED_RESULTS)]
    return set(gone["license_number"].astype("int64"))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw download the same way training did."""
    clean = clean_inspections(raw)
    # features.py reads ids as numbers from the saved CSV, so match that here.
    clean["inspection_id"] = pd.to_numeric(clean["inspection_id"], errors="coerce")
    clean = clean.dropna(subset=["inspection_id"])
    clean["inspection_id"] = clean["inspection_id"].astype("int64")
    return clean.reset_index(drop=True)


def model_violation_codes(feature_cols: list[str]) -> list[str]:
    """The violation numbers the trained model tracks (read from its column names)."""
    return [c.removeprefix("past_viol_") for c in feature_cols if c.startswith("past_viol_")]


# ---------- Predict ----------

def build_snapshot(clean: pd.DataFrame, as_of: pd.Timestamp, model, feature_cols: list[str],
                   closed: set[int] | None = None) -> pd.DataFrame:
    """
    Score every active restaurant as if it had a routine inspection on `as_of`.

    Only inspections BEFORE `as_of` are used. Each restaurant gets one made-up
    "pending" row dated `as_of`, and the normal feature code runs on top of it,
    so live features are built by exactly the same code as the training features.
    """
    history = clean[clean["inspection_date"] < as_of]
    if history.empty:
        raise SystemExit(f"No inspections before {as_of:%Y-%m-%d}, so there is nothing to score.")

    history = history.sort_values(["license_number", "inspection_date", "inspection_id"])
    latest = history.groupby("license_number").last()   # last known value of each column
    active = latest[latest["inspection_date"] >= as_of - pd.Timedelta(days=ACTIVE_DAYS)]
    if closed:
        active = active[~active.index.isin(closed)]

    pending = pd.DataFrame({
        "inspection_id": -(np.arange(len(active)) + 1),   # negative ids mark the made-up rows
        "license_number": active.index.to_numpy(),
        "dba_name": active["dba_name"].to_numpy(),
        "latitude": active["latitude"].to_numpy(),
        "longitude": active["longitude"].to_numpy(),
        "risk_level": active["risk_level"].to_numpy(),
        "inspection_date": as_of,
        "month": as_of.month,
        "inspection_category": "routine",
        "results": None,
        "num_violations": 0,
        "violation_codes": "",
        "inspector_comments": "",
        "new_food_code": 1,
    })
    cols = list(pending.columns)
    combined = pd.concat([history[cols], pending], ignore_index=True)
    feats, _, _ = build_features(combined, top_codes=model_violation_codes(feature_cols))

    snap = feats[feats["inspection_id"] < 0].copy()
    snap["risk_score"] = model.predict_proba(snap[feature_cols])[:, 1]
    snap["risk_percentile"] = snap["risk_score"].rank(pct=True)
    snap = snap.sort_values("risk_score", ascending=False, kind="stable")
    return snap[SNAPSHOT_COLS].reset_index(drop=True)


def save_snapshot(snap: pd.DataFrame, path: Path) -> None:
    out = snap.copy()
    out["risk_score"] = out["risk_score"].round(5)
    out["risk_percentile"] = out["risk_percentile"].round(5)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_snapshots(snap_dir: Path) -> dict[pd.Timestamp, pd.DataFrame]:
    return {pd.Timestamp(p.stem): pd.read_csv(p) for p in sorted(snap_dir.glob("*.csv"))}


# ---------- Score ----------

def score_new_inspections(clean: pd.DataFrame, snapshots: dict[pd.Timestamp, pd.DataFrame],
                          outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Match each new routine inspection to the latest snapshot saved BEFORE the
    day it happened. An inspection is scored once and never rescored.
    """
    if not snapshots:
        return outcomes
    dates = np.array(sorted(snapshots), dtype="datetime64[ns]")

    new = clean[(clean["inspection_category"] == "routine")
                & (clean["inspection_date"] > dates[0])
                & ~clean["inspection_id"].isin(outcomes["inspection_id"])].copy()
    if new.empty:
        return outcomes

    # Index of the last snapshot strictly before each inspection date
    idx = np.searchsorted(dates, new["inspection_date"].to_numpy(dtype="datetime64[ns]"), side="left") - 1
    new["snapshot_date"] = pd.to_datetime(dates[idx])
    new["lead_days"] = (new["inspection_date"] - new["snapshot_date"]).dt.days
    new = new[new["lead_days"] <= MAX_LEAD_DAYS]

    scored = []
    for snap_date, group in new.groupby("snapshot_date"):
        snap = snapshots[pd.Timestamp(snap_date)]
        scored.append(group.merge(snap, on="license_number", how="inner"))   # no earlier score -> skipped
    if not scored:
        return outcomes

    rows = pd.concat(scored, ignore_index=True)[OUTCOME_COLS]
    out = rows if outcomes.empty else pd.concat([outcomes, rows], ignore_index=True)
    return out.sort_values(["inspection_date", "inspection_id"]).reset_index(drop=True)


def load_outcomes(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["inspection_date", "snapshot_date"])
    return pd.DataFrame(columns=OUTCOME_COLS)


# ---------- Report ----------

def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC: the chance a failing restaurant got a higher score than a passing one."""
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarize(outcomes: pd.DataFrame) -> dict:
    """All the numbers in the report, worked out from the outcomes table."""
    y = outcomes["failed"].to_numpy(dtype=int)
    scores = outcomes["risk_score"].to_numpy(dtype=float)
    pct = outcomes["risk_percentile"]
    tier = np.select([pct > 0.8, pct > 0.5], ["High", "Medium"], default="Low")
    tiers = {t: (int((tier == t).sum()), int(y[tier == t].sum())) for t in ["High", "Medium", "Low"]}

    base_flag = (outcomes["last_routine_failed"] == 1).to_numpy()
    k = int(base_flag.sum())
    top_k = np.argsort(-scores, kind="stable")[:k]
    return {
        "n": len(y), "fails": int(y.sum()), "tiers": tiers, "k": k,
        "base_caught": int(y[base_flag].sum()), "model_caught": int(y[top_k].sum()),
        "auc": auc(y, scores) if len(y) else float("nan"),
        "avg_lead": float(outcomes["lead_days"].mean()) if len(y) else float("nan"),
    }


def write_report(outcomes: pd.DataFrame, snapshots: dict, path: Path, replay: bool = False) -> None:
    s = summarize(outcomes)
    lines = ["# Live Track Record", ""]
    if replay:
        lines += ["**This is a replay of past data for testing. It is not the live record.**", ""]
    lines += [
        "The model in `models/gradient_boosting.joblib` is frozen. Every week it scores each active "
        "restaurant, and the scores are committed to this repo before any of those restaurants are "
        "inspected. When an inspection result shows up later, it is compared to the score saved earlier.",
        "",
    ]
    if not snapshots:
        lines += ["No predictions have been saved yet.", ""]
    else:
        first, last = min(snapshots), max(snapshots)
        lines += [
            f"- **Prediction runs:** {len(snapshots)} ({first:%b %d, %Y} to {last:%b %d, %Y})",
            f"- **Restaurants scored in the latest run:** {len(snapshots[last]):,}",
            f"- **Inspections checked against an earlier prediction:** {s['n']:,} ({s['fails']:,} failed)",
            "",
        ]

    if s["n"] == 0:
        lines += ["No inspections have been checked yet. Results appear here once restaurants scored "
                  "in an earlier run get inspected.", ""]
    else:
        if s["n"] < MIN_SCORED or s["fails"] < MIN_FAILS:
            lines += [f"**Early numbers.** With fewer than {MIN_SCORED} inspections or {MIN_FAILS} "
                      "failures checked, these results can still swing a lot from week to week.", ""]
        lines += ["## Did higher-risk restaurants fail more often?", "",
                  "| Risk tier when predicted | Inspections | Failed | Failure rate |", "|---|---|---|---|"]
        for name, (n, f) in s["tiers"].items():
            lines.append(f"| {name} | {n:,} | {f:,} | {'—' if n == 0 else f'{f / n:.1%}'} |")
        lines += ["", "High = top 20% of scores in that week's run, medium = next 30%, low = bottom 50%.", "",
                  "## Model vs. the simple rule", ""]
        if s["k"] == 0:
            lines += ["None of the checked restaurants had failed their last routine inspection, "
                      "so there is no baseline to compare with yet.", ""]
        else:
            lines += [f"The simple rule flags a restaurant if it failed its last routine inspection. It flagged "
                      f"**{s['k']:,}** of the checked inspections and caught **{s['base_caught']:,}** failures. "
                      f"The model's top **{s['k']:,}** scores caught **{s['model_caught']:,}**.", ""]
        lines += [f"- **ROC AUC:** {s['auc']:.3f}" if not np.isnan(s["auc"]) else "- **ROC AUC:** not available yet",
                  f"- **Average days between prediction and inspection:** {s['avg_lead']:.1f}", ""]

    lines += ["## How to check this yourself", "",
              "Each file in `live/snapshots/` is named for the day it was made. The git history shows when it "
              "was committed. `live/outcomes.csv` lists every checked inspection with its date, its result, "
              "and the snapshot its score came from. The snapshot date is always before the inspection date.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------- Run ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=None,
                        help="use this raw CSV instead of downloading the newest data")
    parser.add_argument("--as-of", default=None, help="prediction date, YYYY-MM-DD (default: today)")
    parser.add_argument("--out-dir", default=None,
                        help="folder for snapshots and outcomes (default: live/)")
    parser.add_argument("--token", default=os.environ.get("SOCRATA_APP_TOKEN") or None,
                        help="optional Socrata app token (speeds up downloads)")
    args = parser.parse_args()

    today = pd.Timestamp.today().normalize()
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else today
    out_dir = Path(args.out_dir).resolve() if args.out_dir else LIVE_DIR
    replay = out_dir != LIVE_DIR
    # The live record only means something if every snapshot is dated the day it was made.
    if not replay and as_of != today:
        raise SystemExit("The live record only accepts today's date. To replay a past date, "
                         "add --out-dir with a different folder (for example --out-dir replay).")
    report_path = out_dir / "track_record.md" if replay else REPORT_PATH
    snap_dir, outcomes_path = out_dir / "snapshots", out_dir / "outcomes.csv"

    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found. Run `python src/train.py` first.")
    saved = joblib.load(MODEL_PATH)
    model, feature_cols = saved["model"], saved["features"]

    raw = pd.read_csv(args.raw, dtype=str) if args.raw else download_all(args.token)
    clean = prepare(raw)
    closed = closed_licenses(raw)
    newest = clean["inspection_date"].max()
    print(f"{len(clean):,} restaurant inspections, newest on {newest:%Y-%m-%d}")
    if (as_of - newest).days > STALE_DAYS:
        print(f"WARNING: the newest inspection is more than {STALE_DAYS} days before {as_of:%Y-%m-%d}. "
              "The data may be out of date.")

    # 1. Score inspections that happened after earlier snapshots
    snapshots = load_snapshots(snap_dir)
    outcomes = load_outcomes(outcomes_path)
    before = len(outcomes)
    outcomes = score_new_inspections(clean, snapshots, outcomes)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(outcomes_path, index=False)
    print(f"Checked {len(outcomes) - before:,} new inspections against earlier predictions "
          f"({len(outcomes):,} total)")

    # 2. Save today's predictions (a snapshot is never overwritten)
    snap_path = snap_dir / f"{as_of:%Y-%m-%d}.csv"
    if snap_path.exists():
        print(f"{snap_path.name} already exists, so it was left alone.")
    else:
        snap = build_snapshot(clean, as_of, model, feature_cols, closed)
        save_snapshot(snap, snap_path)
        snapshots = load_snapshots(snap_dir)
        print(f"Saved risk scores for {len(snap):,} restaurants to {snap_path}")

    write_report(outcomes, snapshots, report_path, replay=replay)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
