"""Tests for the LLM evaluation (src/llm_eval.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_eval  # noqa: E402

GROUPS = pd.DataFrame({
    "code": [38, 22, 10, 55, 49, 2, 39],
    "title": ["INSECTS, RODENTS, & ANIMALS NOT PRESENT", "PROPER COLD HOLDING TEMPERATURES",
              "ADEQUATE HANDWASHING SINKS PROPERLY SUPPLIED AND ACCESSIBLE",
              "PHYSICAL FACILITIES INSTALLED, MAINTAINED & CLEAN", "NON-FOOD/FOOD CONTACT SURFACES CLEAN",
              "CITY OF CHICAGO FOOD SERVICE SANITATION CERTIFICATE",
              "CONTAMINATION PREVENTED DURING FOOD PREPARATION, STORAGE & DISPLAY"],
    "group": ["pests", "temperature", "hand hygiene", "facilities", "cleaning",
              "management & paperwork", "food protection"],
})
VALID = sorted(GROUPS["group"])
TEMPLATES = {
    38: "observed {n} mouse droppings along the wall base in the rear storage room",
    22: "cold holding cooler at {n} degrees, foods discarded, must hold at 41 or below",
    10: "no soap or paper towels at the hand sink near the prep line number {n}",
    55: "floors under the cooking line need cleaning and repair, area {n}",
    49: "interior of the reach in cooler shelf {n} has food debris, must clean",
    2: "no certified food manager on duty during inspection, citation {n}",
    39: "raw chicken stored above ready to eat food in walk in cooler shelf {n}",
}


def make_clean(n_per_code: int = 60) -> pd.DataFrame:
    """Fake cleaned inspections, in the same format clean.py writes."""
    rows, iid = [], 1
    dates = pd.date_range("2023-01-01", "2026-06-30", periods=n_per_code)   # about 40% from 2025 on
    for k in range(n_per_code):
        pieces = []
        for code, title in zip(GROUPS["code"], GROUPS["title"]):
            pieces.append(f"{code}. {title} - Comments: {TEMPLATES[code].format(n=k)}")
        pieces.append("60. PREVIOUS CORE VIOLATION CORRECTED - Comments: previous violation corrected ok")
        pieces.append("49. NON-FOOD/FOOD CONTACT SURFACES CLEAN - Comments: dirty")   # too short
        rows.append({"inspection_id": iid, "inspection_date": dates[k].strftime("%Y-%m-%d"),
                     "violations": " | ".join(pieces), "new_food_code": 1})
        iid += 1
    rows.append({"inspection_id": iid, "inspection_date": "2017-01-01", "new_food_code": 0,
                 "violations": "38. RODENTS - Comments: old food code note, should be skipped entirely"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def labeled() -> pd.DataFrame:
    return llm_eval.label_notes(llm_eval.split_notes(make_clean()), GROUPS)


@pytest.fixture(scope="module")
def test_set(labeled) -> pd.DataFrame:
    return llm_eval.make_test_set(labeled, per_group=5)


# ------------------------------------------------------------------ notes ----

def test_split_notes_reads_code_and_comment_only():
    notes = llm_eval.split_notes(make_clean(2))
    assert set(notes["code"]) == {2, 10, 22, 38, 39, 49, 55, 60}       # old food code row skipped
    first = notes[(notes["inspection_id"] == 1) & (notes["code"] == 38)]["note"].iloc[0]
    assert first == TEMPLATES[38].format(n=0)
    assert not notes["note"].str.contains("INSECTS, RODENTS").any()   # the official title is not in the note


def test_label_notes_drops_unlisted_codes_and_short_notes(labeled):
    assert 60 not in set(labeled["code"])
    assert (labeled["note"].str.len() >= llm_eval.MIN_CHARS).all()
    assert set(labeled["group"]) == set(VALID)


def test_test_set_is_recent_balanced_and_unique(test_set):
    assert (test_set["inspection_date"] >= llm_eval.TEST_START).all()
    assert test_set["group"].value_counts().eq(5).all()
    assert test_set["note"].is_unique
    assert list(test_set["note_id"]) == list(range(1, len(test_set) + 1))


def test_test_set_is_the_same_every_time(labeled, test_set):
    pd.testing.assert_frame_equal(test_set, llm_eval.make_test_set(labeled, per_group=5))


def test_training_notes_are_older_and_never_copy_a_test_note(labeled, test_set):
    copied = labeled.iloc[[0]].assign(note=test_set["note"].iloc[0],
                                      inspection_date=pd.Timestamp("2023-02-01"))
    train = llm_eval.training_notes(pd.concat([labeled, copied], ignore_index=True), test_set)
    assert (train["inspection_date"] < llm_eval.TEST_START).all()
    assert not train["note"].isin(test_set["note"]).any()


def test_too_few_recent_notes_stops_with_a_message(labeled):
    with pytest.raises(SystemExit, match="Not enough recent notes"):
        llm_eval.make_test_set(labeled, per_group=10_000)


# -------------------------------------------------------------- baselines ----

def test_keyword_rules():
    notes = pd.Series(["Mice droppings found", "Cooler at 50 degrees", "no soap at hand sink",
                       "certified manager absent", "wall needs paint"])
    assert llm_eval.keyword_predict(notes) == ["pests", "temperature", "hand hygiene",
                                               "management & paperwork", llm_eval.KEYWORD_FALLBACK]
    assert llm_eval.keyword_coverage(notes) == pytest.approx(0.8)


def test_strip_numbers():
    assert llm_eval.strip_numbers("PRIORITY 7-38-005 at 45F") == "PRIORITY  - -  at  F"


def test_trained_model_learns_the_fake_notes(labeled, test_set):
    train = llm_eval.training_notes(labeled, test_set)
    model = llm_eval.train_model(train)
    assert (model.predict(test_set["note"]) == test_set["group"]).mean() == 1.0


# -------------------------------------------------------------------- LLM ----

def test_prompt_lists_every_category_and_every_note(test_set):
    batch = test_set.head(3)
    system, user = llm_eval.build_prompt(batch, GROUPS)
    for group in VALID:
        assert f'"{group}"' in system["content"]
    for row in batch.itertuples():
        assert f"Note {row.note_id}: {row.note}" in user["content"]
    assert str(batch["group"].iloc[0]) not in user["content"]          # the answer is never in the note text


@pytest.mark.parametrize("content", [
    '{"labels": [{"id": 1, "category": "pests"}, {"id": 2, "category": "Management and Paperwork"}]}',
    '```json\n{"labels": [{"id": 1, "category": "PESTS"}, {"id": 2, "category": "management & paperwork"}]}\n```',
    'Here you go:\n[{"id": 1, "category": "pests"}, {"id": 2, "category": "management & paperwork"}]',
])
def test_parse_response_formats(content):
    assert llm_eval.parse_response(content, [1, 2], VALID) == {1: "pests", 2: "management & paperwork"}


def test_parse_response_ignores_bad_answers():
    content = json.dumps({"labels": [{"id": 1, "category": "not a category"}, {"id": 9, "category": "pests"},
                                     {"id": "x"}, {"id": 2, "category": "pests"}, {"id": 2, "category": "cleaning"}]})
    assert llm_eval.parse_response(content, [1, 2], VALID) == {2: "pests"}
    assert llm_eval.parse_response("I cannot help with that", [1], VALID) == {}
    assert llm_eval.parse_response("", [1], VALID) == {}


class FakeClient:
    """Answers correctly, except it skips the last note of every batch and once returns junk."""

    def __init__(self, truth: dict[int, str]):
        self.truth, self.calls = truth, 0

    def chat(self, messages):
        self.calls += 1
        ids = [int(part.split(":")[0]) for part in messages[1]["content"].split("Note ")[1:]]
        if self.calls == 3:                                           # the second batch request
            return "sorry"
        keep = ids[:-1] if len(ids) > 1 else ids
        return json.dumps({"labels": [{"id": i, "category": self.truth[i]} for i in keep]})


def test_run_llm_answers_everything_retries_and_resumes(test_set, tmp_path):
    truth = dict(zip(test_set["note_id"], test_set["group"]))
    path = tmp_path / "predictions.jsonl"
    client = FakeClient(truth)
    answers = llm_eval.run_llm(test_set, GROUPS, client, "fake/model", batch_size=10, path=path)
    assert answers == truth                                           # every note, all correct
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == len(test_set) and {r["prompt_version"] for r in lines} == {llm_eval.PROMPT_VERSION}
    assert any(r["batch"] == "single" for r in lines)                 # missing notes were asked again

    again = FakeClient(truth)
    assert llm_eval.run_llm(test_set, GROUPS, again, "fake/model", path=path) == truth
    assert again.calls == 0                                           # nothing asked twice


def test_note_with_no_usable_answer_is_scored_wrong(test_set, tmp_path):
    class Useless:
        calls = 0

        def chat(self, messages):
            self.calls += 1
            return "no idea"
    answers = llm_eval.run_llm(test_set.head(2), GROUPS, Useless(), "fake/model", path=tmp_path / "p.jsonl")
    assert set(answers.values()) == {"no answer"}


def test_answers_from_an_older_prompt_are_not_reused(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"note_id": 1, "category": "pests", "prompt_version": "old"}) + "\n"
                    + json.dumps({"note_id": 2, "category": "cleaning",
                                  "prompt_version": llm_eval.PROMPT_VERSION}) + "\n")
    assert llm_eval.load_predictions(path) == {2: "cleaning"}


# ---------------------------------------------------------------- scoring ----

def test_score_matches_sklearn():
    from sklearn.metrics import f1_score
    truth = pd.Series(["pests", "pests", "cleaning", "cleaning", "facilities", "facilities"])
    pred = ["pests", "cleaning", "cleaning", "cleaning", "facilities", "pests"]
    groups = ["cleaning", "facilities", "pests"]
    s = llm_eval.score(truth, pred, groups)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["macro_f1"] == pytest.approx(f1_score(truth, pred, average="macro"))
    assert s["f1_cleaning"] == pytest.approx(0.8)
    assert s["accuracy_low"] <= s["accuracy"] <= s["accuracy_high"]


def test_paired_difference():
    truth = ["a", "a", "b", "b"]
    d, lo, hi = llm_eval.paired_difference(truth, ["a", "a", "b", "b"], ["a", "b", "b", "a"])
    assert d == pytest.approx(0.5)
    assert lo <= d <= hi
    same = llm_eval.paired_difference(truth, ["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert same == (0.0, 0.0, 0.0)


def test_report_wording_follows_the_numbers(test_set, tmp_path):
    groups = sorted(test_set["group"].unique())

    def row(method, short, kind, acc, lo, hi, diff=(np.nan, np.nan, np.nan)):
        return {"method": method, "short": short, "kind": kind, "n": 35, "accuracy": acc,
                "accuracy_low": lo, "accuracy_high": hi, "macro_f1": acc, **{f"f1_{g}": acc for g in groups},
                "diff_vs_trained": diff[0], "diff_low": diff[1], "diff_high": diff[2]}
    base = [row("Keyword rules (the risk model's word lists)", "Keyword rules", "keyword", 0.5, 0.4, 0.6),
            row("Trained model (TF-IDF + logistic regression)", "Trained model", "trained", 0.95, 0.92, 0.98),
            row("Trained model, numbers removed from notes", "Trained, no numbers", "trained", 0.94, 0.91, 0.97)]
    info = {"n_train": 1000, "batch_size": 10, "excluded_share": 0.05, "keyword_coverage": 0.6, "llm_usage": {}}
    cases = [([], "has not been run yet"),
             # Separate intervals overlap here, but the paired difference is clearly below zero
             ([row("LLM, zero-shot (m)", "m", "llm", 0.90, 0.87, 0.94, (-0.05, -0.08, -0.02))],
              "The trained model wins"),
             ([row("LLM, zero-shot (m)", "m", "llm", 0.94, 0.91, 0.97, (-0.01, -0.03, 0.01))], "are close"),
             ([row("LLM, zero-shot (m)", "m", "llm", 0.99, 0.985, 1.0, (0.04, 0.02, 0.06))], "The LLM wins")]
    for extra, expected in cases:
        path = llm_eval.write_report(pd.DataFrame(base + extra), test_set, info, tmp_path / "r.md")
        assert expected in path.read_text()
