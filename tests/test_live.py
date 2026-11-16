"""Tests for the live prediction pipeline (src/live.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import live  # noqa: E402
from features import CODE_CHANGE_DATE, build_features  # noqa: E402

VIOLATIONS = [
    "55. PHYSICAL FACILITIES INSTALLED, MAINTAINED & CLEAN - Comments: mice droppings under the sink",
    "47. FOOD & NON-FOOD CONTACT SURFACES CLEANABLE - Comments: torn gasket on cooler door",
    "10. ADEQUATE HANDWASHING SINKS PROPERLY SUPPLIED - Comments: no soap at hand sink",
    "38. INSECTS, RODENTS, & ANIMALS NOT PRESENT - Comments: live roach seen near the drain",
]


def make_raw(n_licenses: int = 60, seed: int = 0) -> pd.DataFrame:
    """A fake raw download: a few years of inspections for a handful of restaurants."""
    rng = np.random.default_rng(seed)
    rows, next_id = [], 1000
    for lic in range(1, n_licenses + 1):
        risky = rng.random() < 0.4
        date = pd.Timestamp("2021-01-04") + pd.Timedelta(days=int(rng.integers(0, 200)))
        while date < pd.Timestamp("2026-06-30"):
            fail = rng.random() < (0.5 if risky else 0.1)
            n_viol = int(rng.integers(2, 5)) if (risky or fail) else int(rng.integers(0, 2))
            picked = rng.choice(len(VIOLATIONS), size=n_viol, replace=False)
            rows.append({
                "inspection_id": str(next_id), "dba_name": f"PLACE {lic}", "aka_name": f"PLACE {lic}",
                "license_": str(lic), "facility_type": "Restaurant", "risk": "Risk 1 (High)",
                "address": f"{lic} W MAIN ST", "zip": "60601",
                "inspection_date": f"{date:%Y-%m-%d}T00:00:00.000",
                "inspection_type": "Canvass" if rng.random() < 0.8 else "Complaint",
                "results": "Fail" if fail else ("Pass w/ Conditions" if rng.random() < 0.3 else "Pass"),
                "violations": " | ".join(VIOLATIONS[i] for i in picked),
                "latitude": "41.88", "longitude": "-87.63",
            })
            next_id += 1
            date += pd.Timedelta(days=int(rng.integers(60, 240)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="module")
def clean(raw) -> pd.DataFrame:
    return live.prepare(raw)


@pytest.fixture(scope="module")
def trained(clean):
    """A small model trained the same way train.py does, on the fake data."""
    feats, cols, _ = build_features(clean)
    rows = feats[(feats["is_routine"] == 1) & (feats["inspection_date"] >= CODE_CHANGE_DATE)]
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(rows[cols], rows["failed"])
    return model, cols


def test_live_scores_match_training_features(clean, trained):
    """A live score must equal what the training feature code gives for the same inspection."""
    model, cols = trained
    feats, _, _ = build_features(clean, top_codes=live.model_violation_codes(cols))
    routine = feats[(feats["is_routine"] == 1) & (feats["prior_inspections"] > 0)]
    checked = 0
    for _, row in routine.sample(25, random_state=1).iterrows():
        snap = live.build_snapshot(clean, row["inspection_date"], model, cols)
        got = snap.loc[snap["license_number"] == row["license_number"], "risk_score"]
        if got.empty:   # not active on that date
            continue
        want = model.predict_proba(row[cols].to_frame().T.astype(float))[0, 1]
        assert got.iloc[0] == pytest.approx(want)
        checked += 1
    assert checked >= 15


def test_snapshot_ignores_everything_on_or_after_its_date(clean, trained):
    model, cols = trained
    as_of = pd.Timestamp("2025-03-01")
    full = live.build_snapshot(clean, as_of, model, cols)
    past_only = live.build_snapshot(clean[clean["inspection_date"] < as_of], as_of, model, cols)
    pd.testing.assert_frame_equal(full, past_only)


def test_snapshot_skips_closed_and_inactive_restaurants(clean, trained):
    model, cols = trained
    as_of = pd.Timestamp("2026-07-15")
    everyone = live.build_snapshot(clean, as_of, model, cols)
    assert everyone["license_number"].is_unique
    without_7 = live.build_snapshot(clean, as_of, model, cols, closed={7})
    assert 7 in set(everyone["license_number"]) and 7 not in set(without_7["license_number"])

    # A restaurant last inspected more than three years ago is no longer scored
    old = clean[clean["license_number"] == 1].head(1).copy()
    old["license_number"], old["inspection_id"] = 999, 5
    old["inspection_date"] = as_of - pd.Timedelta(days=live.ACTIVE_DAYS + 10)
    with_old = live.build_snapshot(pd.concat([clean, old], ignore_index=True), as_of, model, cols)
    assert 999 not in set(with_old["license_number"])


def test_closed_licenses_reads_the_latest_record(raw):
    extra = raw[raw["license_"] == "3"].tail(1).copy()
    extra["inspection_date"], extra["results"], extra["inspection_id"] = "2026-07-10T00:00:00.000", "Out of Business", "1"
    assert 3 in live.closed_licenses(pd.concat([raw, extra], ignore_index=True))
    assert 3 not in live.closed_licenses(raw)


def test_inspections_are_matched_to_an_earlier_snapshot_only(clean, trained):
    model, cols = trained
    d1, d2 = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-16")
    snapshots = {d: live.build_snapshot(clean, d, model, cols) for d in (d1, d2)}
    out = live.score_new_inspections(clean, snapshots, live.load_outcomes(Path("does-not-exist.csv")))

    assert len(out) > 0
    assert (out["snapshot_date"] < out["inspection_date"]).all()
    assert (out["inspection_date"] > d1).all()                     # nothing from before the first run
    assert (out["lead_days"] <= live.MAX_LEAD_DAYS).all()
    late = out[out["inspection_date"] > d2]
    assert (late["snapshot_date"] == d2).all()                     # always the newest earlier snapshot
    assert (out[out["inspection_date"] <= d2]["snapshot_date"] == d1).all()   # same-day counts as not earlier
    assert set(out["results"]) <= {"Pass", "Pass w/ Conditions", "Fail"}

    # The saved score is the one from that snapshot
    row = out.iloc[0]
    snap = snapshots[row["snapshot_date"]]
    assert row["risk_score"] == snap.loc[snap["license_number"] == row["license_number"], "risk_score"].iloc[0]


def test_an_inspection_is_scored_only_once(clean, trained):
    model, cols = trained
    snapshots = {pd.Timestamp("2026-03-02"): live.build_snapshot(clean, pd.Timestamp("2026-03-02"), model, cols)}
    once = live.score_new_inspections(clean, snapshots, live.load_outcomes(Path("does-not-exist.csv")))
    twice = live.score_new_inspections(clean, snapshots, once)
    assert len(twice) == len(once)
    assert twice["inspection_id"].is_unique


def test_summary_numbers():
    outcomes = pd.DataFrame({
        "failed":              [1,    1,    0,    0,    1,    0],
        "risk_score":          [0.9,  0.8,  0.7,  0.3,  0.2,  0.1],
        "risk_percentile":     [0.95, 0.85, 0.60, 0.40, 0.30, 0.10],
        "last_routine_failed": [1,    0,    1,    0,    0,    np.nan],
        "lead_days":           [3,    5,    7,    2,    4,    3],
    })
    s = live.summarize(outcomes)
    assert (s["n"], s["fails"], s["k"]) == (6, 3, 2)
    assert s["tiers"] == {"High": (2, 2), "Medium": (1, 0), "Low": (3, 1)}
    assert s["base_caught"] == 1      # the two baseline flags held one failure
    assert s["model_caught"] == 2     # the model's top two scores held two
    assert s["auc"] == pytest.approx(7 / 9)
    assert s["avg_lead"] == pytest.approx(4.0)


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["live.py", *args])
    live.main()


def test_live_record_refuses_a_backdated_snapshot(monkeypatch):
    with pytest.raises(SystemExit, match="only accepts today's date"):
        run_main(monkeypatch, "--as-of", "2020-01-01")


def test_replay_end_to_end(raw, trained, tmp_path, monkeypatch):
    model, cols = trained
    model_path, raw_path, out_dir = tmp_path / "model.joblib", tmp_path / "raw.csv", tmp_path / "replay"
    joblib.dump({"model": model, "features": cols}, model_path)
    raw.to_csv(raw_path, index=False)
    monkeypatch.setattr(live, "MODEL_PATH", model_path)

    run_main(monkeypatch, "--raw", str(raw_path), "--as-of", "2026-03-02", "--out-dir", str(out_dir))
    assert (out_dir / "snapshots" / "2026-03-02.csv").exists()
    assert len(pd.read_csv(out_dir / "outcomes.csv")) == 0      # first run has nothing to check yet

    first = (out_dir / "snapshots" / "2026-03-02.csv").read_text()
    run_main(monkeypatch, "--raw", str(raw_path), "--as-of", "2026-04-06", "--out-dir", str(out_dir))
    outcomes = pd.read_csv(out_dir / "outcomes.csv", parse_dates=["inspection_date", "snapshot_date"])
    assert len(outcomes) > 0 and outcomes["inspection_id"].is_unique
    assert (outcomes["snapshot_date"] < outcomes["inspection_date"]).all()
    assert (out_dir / "snapshots" / "2026-03-02.csv").read_text() == first   # old snapshots never change

    report = (out_dir / "track_record.md").read_text()
    assert "replay" in report and "Prediction runs:** 2" in report
