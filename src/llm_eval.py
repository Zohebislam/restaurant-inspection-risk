"""
Step 7: Can an LLM read an inspector's note and tell what kind of violation it is?

Run from the project folder:
    python src/llm_eval.py                     # baselines + the LLM, then the report
    python src/llm_eval.py --skip-llm          # baselines and report only

Every violation in the data comes with an official code (for example 38 =
"Insects, rodents & animals") and the inspector's free-text note. That makes a
ready-made answer key: give a method only the note, and check whether it picks
the right category. Nobody has to label anything by hand.

Three methods are compared on the same 280 recent notes (40 per category):
  1. Keyword rules: the same word lists the risk model uses (src/features.py)
  2. Trained model: TF-IDF + logistic regression, trained on notes from before 2025
  3. LLM: a Gemini model, zero-shot (category names and official titles only,
     no training examples), called through a local OmniRoute gateway

The LLM is reached through OmniRoute's OpenAI-compatible endpoint. Set the key
in your terminal first (it is never written to a file in this repo):
    export OMNIROUTE_API_KEY=$(cat ~/.omniroute_key)

LLM answers are saved to reports/llm_eval/predictions_<model>.jsonl as they
arrive, so a run that stops early picks up where it left off, and anyone can
rebuild the report without calling the LLM again.

Outputs:
  reports/llm_eval/test_notes.csv     the fixed test set
  reports/llm_eval/scores.csv         accuracy and F1 for every method
  reports/llm_eval/note_predictions.csv  every method's answer for every test note
  reports/llm_eval.md                 the write-up
  reports/figures/12_llm_eval.png     chart
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "inspections_clean.csv"
GROUPS_PATH = ROOT / "data" / "reference" / "violation_groups.csv"
OUT_DIR = ROOT / "reports" / "llm_eval"
TEST_PATH = OUT_DIR / "test_notes.csv"
NOTE_PREDICTIONS_PATH = OUT_DIR / "note_predictions.csv"
SCORES_PATH = OUT_DIR / "scores.csv"
REPORT_PATH = ROOT / "reports" / "llm_eval.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "12_llm_eval.png"

DEFAULT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_BASE_URL = "http://localhost:20128/v1"
TEST_START = pd.Timestamp("2025-01-01")   # test notes come from here on; training notes from before
PER_GROUP = 40                            # test notes per category
MIN_CHARS = 20                            # skip notes too short to classify
SEED = 42
PROMPT_VERSION = "v1"                     # change this if the prompt changes, so old answers are not reused

VIOLATION_RE = re.compile(r"^\s*(\d+)\.\s*(.*?)(?:\s*-\s*Comments:\s*(.*))?$", re.DOTALL)

# The risk model's keyword lists (src/features.py), turned into one guess per
# note: the first list that matches wins. If none match, guess "facilities",
# the most common category.
KEYWORD_RULES = [
    ("pests", r"rodent|mouse|mice|rat |rats|droppings|roach|insect|flies|fly |gnat"),
    ("temperature", r"temperature|temp\.|degrees|cold holding|hot holding"),
    ("hand hygiene", r"hand ?wash|hand sink|handsink|soap|paper towel"),
    ("facilities", r"no hot water|hot water|sewage|backup|back up|drain"),
    ("cleaning", r"mold|mildew|slime"),
    ("food protection", r"raw|cross[- ]contam|uncovered"),
    ("management & paperwork", r"certificate|certified|food handler|manager"),
]
KEYWORD_FALLBACK = "facilities"


# ---------------------------------------------------------------- the notes --

def load_groups(path: Path = GROUPS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"code": int, "title": str, "group": str})


def split_notes(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per violation: inspection, date, official code, and the inspector's note."""
    df = clean[clean["new_food_code"] == 1] if "new_food_code" in clean else clean
    rows = []
    for iid, date, text in zip(df["inspection_id"], df["inspection_date"], df["violations"].fillna("")):
        for piece in str(text).split("|"):
            match = VIOLATION_RE.match(piece.strip())
            if match:
                rows.append((iid, date, int(match.group(1)), (match.group(3) or "").strip()))
    notes = pd.DataFrame(rows, columns=["inspection_id", "inspection_date", "code", "note"])
    notes["inspection_date"] = pd.to_datetime(notes["inspection_date"])
    return notes


def label_notes(notes: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Attach each note's category. Codes not in the list (rare or administrative) are dropped."""
    out = notes.merge(groups[["code", "group"]], on="code", how="inner")
    return out[out["note"].str.len() >= MIN_CHARS].reset_index(drop=True)


def make_test_set(labeled: pd.DataFrame, per_group: int = PER_GROUP, seed: int = SEED) -> pd.DataFrame:
    """40 notes per category from 2025 on. Copy-pasted notes count once."""
    pool = labeled[labeled["inspection_date"] >= TEST_START].drop_duplicates("note")
    short = pool["group"].value_counts()
    if (short < per_group).any():
        raise SystemExit(f"Not enough recent notes for: {', '.join(short[short < per_group].index)}")
    test = pool.groupby("group", group_keys=False).sample(per_group, random_state=seed)
    test = test.sort_values(["group", "inspection_date", "inspection_id"]).reset_index(drop=True)
    test.insert(0, "note_id", np.arange(1, len(test) + 1))
    return test


def training_notes(labeled: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Notes from before 2025. Any note word-for-word identical to a test note is removed,
    because inspectors reuse text and the trained model would just memorize it."""
    train = labeled[labeled["inspection_date"] < TEST_START]
    return train[~train["note"].isin(set(test["note"]))].reset_index(drop=True)


# --------------------------------------------------------------- baselines --

def keyword_predict(notes: pd.Series) -> list[str]:
    out = []
    for text in notes.str.lower():
        out.append(next((g for g, pattern in KEYWORD_RULES if re.search(pattern, text)), KEYWORD_FALLBACK))
    return out


def keyword_coverage(notes: pd.Series) -> float:
    """Share of notes where at least one keyword list matched (the rest got the fallback)."""
    return float(np.mean([any(re.search(p, t) for _, p in KEYWORD_RULES) for t in notes.str.lower()]))


def strip_numbers(text: str) -> str:
    """Remove digits, so ordinance numbers like 7-38-005 cannot give the answer away."""
    return re.sub(r"\d+", " ", text)


def train_model(train: pd.DataFrame, numbers: bool = True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    text = train["note"] if numbers else train["note"].map(strip_numbers)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200_000),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )
    return model.fit(text, train["group"])


# --------------------------------------------------------------------- LLM --

def build_prompt(batch: pd.DataFrame, groups: pd.DataFrame) -> list[dict]:
    """Zero-shot: the category names and their official violation titles, nothing else."""
    lines = []
    for group, titles in groups.groupby("group", sort=True)["title"]:
        lines.append(f'- "{group}": ' + "; ".join(t.title() for t in titles))
    system = (
        "You classify Chicago restaurant health inspection notes. Each note was written by an "
        "inspector about one violation. Pick the one category the violation belongs to.\n\n"
        "Categories (each followed by the official violation titles it covers):\n"
        + "\n".join(lines)
        + "\n\nAnswer with JSON only, in this form: "
        '{"labels": [{"id": 1, "category": "pests"}]}. '
        "Include every id exactly once and use the category names exactly as written above."
    )
    user = "\n\n".join(f"Note {row.note_id}: {row.note}" for row in batch.itertuples())
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_category(text, valid: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    key = re.sub(r"[^a-z]+", " ", text.lower().replace("&", " and ")).split()
    for group in valid:
        if key == re.sub(r"[^a-z]+", " ", group.lower().replace("&", " and ")).split():
            return group
    return None


def parse_response(content: str, ids: list[int], valid: list[str]) -> dict[int, str]:
    """Read the model's JSON answer. Returns {note_id: category} for the answers that are usable."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start >= 0 else json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            data = {"labels": json.loads(text[text.find("["):text.rfind("]") + 1])}
        except (json.JSONDecodeError, ValueError):
            return {}
    items = data.get("labels", []) if isinstance(data, dict) else data
    out = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            note_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        category = normalize_category(item.get("category"), valid)
        if note_id in ids and category and note_id not in out:
            out[note_id] = category
    return out


class StopRun(Exception):
    """Raised when there is no point retrying now (bad key, daily limit reached)."""


class OmniRouteClient:
    """Minimal client for an OpenAI-compatible chat endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str, wait: float, timeout: float = 600):
        self.model, self.base_url, self.api_key, self.wait = model, base_url.rstrip("/"), api_key, wait
        self.timeout = timeout
        self._last = 0.0
        self.requests = 0
        self.tokens = {"prompt": 0, "completion": 0}

    def chat(self, messages: list[dict]) -> str:
        import requests

        for attempt in range(6):
            pause = self.wait - (time.time() - self._last)
            if pause > 0:
                time.sleep(pause)
            self._last = time.time()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "temperature": 0},
                    timeout=self.timeout,
                )
            except requests.ConnectionError as err:
                raise StopRun(f"Could not reach OmniRoute at {self.base_url} ({err}). Is it running?")
            except requests.Timeout:
                print(f"    no answer after {self.timeout:.0f}s, trying again...")
                continue
            except requests.RequestException as err:
                print(f"    request failed ({err}), trying again in 15s...")
                time.sleep(15)
                continue
            self.requests += 1
            if resp.status_code in (401, 403):
                raise StopRun("OmniRoute rejected the API key. Run: export OMNIROUTE_API_KEY=$(cat ~/.omniroute_key)")
            if resp.status_code == 429 or resp.status_code >= 500:
                body = resp.text.lower()
                if resp.status_code == 429 and ("per day" in body or "daily" in body or "perday" in body):
                    raise StopRun("The free tier's daily limit is used up. Run this again tomorrow; "
                                  "it will pick up where it stopped.")
                wait = 30 * 2 ** attempt
                print(f"    busy ({resp.status_code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise StopRun(f"OmniRoute returned {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            usage = data.get("usage") or {}
            self.tokens["prompt"] += int(usage.get("prompt_tokens") or 0)
            self.tokens["completion"] += int(usage.get("completion_tokens") or 0)
            return data["choices"][0]["message"]["content"] or ""
        raise StopRun("Gave up after repeated errors. Run this again later; finished answers are saved.")


def predictions_path(model: str) -> Path:
    return OUT_DIR / f"predictions_{re.sub(r'[^A-Za-z0-9.-]+', '_', model)}.jsonl"


def load_predictions(path: Path) -> dict[int, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("prompt_version") == PROMPT_VERSION:
                    out[int(row["note_id"])] = row["category"]
    return out


def run_llm(test: pd.DataFrame, groups: pd.DataFrame, client, model: str,
            batch_size: int = 10, path: Path | None = None) -> dict[int, str]:
    """Ask the LLM about every test note it has not answered yet, saving each answer."""
    path = path or predictions_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_predictions(path)
    valid = sorted(groups["group"].unique())
    todo = test[~test["note_id"].isin(done)]
    if todo.empty:
        print(f"  {model}: all {len(test)} notes already answered")
        return done

    def save(answers: dict[int, str], how: str) -> None:
        with path.open("a") as f:
            for note_id, category in sorted(answers.items()):
                f.write(json.dumps({"note_id": note_id, "category": category, "model": model,
                                    "prompt_version": PROMPT_VERSION, "batch": how}) + "\n")
        done.update(answers)

    batches = [todo.iloc[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  {model}: {len(todo)} notes left, {len(batches)} requests")
    for n, batch in enumerate(batches, start=1):
        ids = batch["note_id"].tolist()
        answers = parse_response(client.chat(build_prompt(batch, groups)), ids, valid)
        save(answers, "batch")
        for note_id in [i for i in ids if i not in answers]:         # ask again one at a time
            single = batch[batch["note_id"] == note_id]
            answer = parse_response(client.chat(build_prompt(single, groups)), [note_id], valid)
            save(answer or {note_id: "no answer"}, "single")
        print(f"    request {n}/{len(batches)} done ({len(done)}/{len(test)} notes answered)")
    return done


# ----------------------------------------------------------------- scoring --

def score(truth: pd.Series, pred, groups: list[str], n_boot: int = 2000, seed: int = SEED) -> dict:
    """Accuracy with a 95% bootstrap interval, macro F1, and F1 for each category."""
    from sklearn.metrics import f1_score

    y, p = np.asarray(truth), np.asarray(pred)
    correct = (y == p).astype(float)
    rng = np.random.default_rng(seed)
    boots = correct[rng.integers(0, len(correct), size=(n_boot, len(correct)))].mean(axis=1)
    per_group = f1_score(y, p, labels=groups, average=None, zero_division=0)
    out = {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "accuracy_low": float(np.quantile(boots, 0.025)),
        "accuracy_high": float(np.quantile(boots, 0.975)),
        "macro_f1": float(f1_score(y, p, labels=groups, average="macro", zero_division=0)),
    }
    out.update({f"f1_{g}": float(v) for g, v in zip(groups, per_group)})
    return out


def paired_difference(truth, pred_a, pred_b, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Accuracy of A minus accuracy of B on the same notes, with a 95% bootstrap interval.
    Resampling notes (not each method separately) uses the fact that both answered the
    same questions, which is a fairer and tighter comparison than two separate intervals."""
    y = np.asarray(truth)
    diff = (np.asarray(pred_a) == y).astype(float) - (np.asarray(pred_b) == y).astype(float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ------------------------------------------------------------------ report --

def write_report(scores: pd.DataFrame, test: pd.DataFrame, info: dict, path: Path = REPORT_PATH) -> Path:
    groups = sorted(test["group"].unique())
    llm = scores[scores["kind"] == "llm"]
    trained = scores[scores["method"] == "Trained model (TF-IDF + logistic regression)"].iloc[0]
    trained_no_num = scores[scores["method"].str.startswith("Trained model, numbers removed")].iloc[0]
    keywords = scores[scores["method"].str.startswith("Keyword rules")].iloc[0]

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    table = ["| Method | Accuracy (95% CI) | Macro F1 |", "|---|---|---|"]
    for r in scores.itertuples():
        table.append(f"| {r.method} | **{pct(r.accuracy)}** ({pct(r.accuracy_low)} to {pct(r.accuracy_high)}) "
                     f"| {r.macro_f1:.3f} |")
    f1_table = ["| Category | " + " | ".join(scores["short"]) + " |", "|---" * (len(scores) + 1) + "|"]
    for g in groups:
        f1_table.append(f"| {g} | " + " | ".join(f"{v:.2f}" for v in scores[f"f1_{g}"]) + " |")

    if llm.empty:
        verdict = ("The LLM has not been run yet. Run `python src/llm_eval.py` with OmniRoute running "
                   "to add it.")
    else:
        best = llm.sort_values("accuracy", ascending=False).iloc[0]
        gap, gap_low, gap_high = best["diff_vs_trained"], best["diff_low"], best["diff_high"]
        paired = (f" Compared note by note on the same {len(test)} notes, the difference is "
                  f"{100 * gap:+.1f} points (95% interval {100 * gap_low:+.1f} to {100 * gap_high:+.1f}).")
        if gap_high < 0:
            verdict = (f"**The trained model wins.** {best['method']} got {pct(best['accuracy'])} with no training "
                       f"examples, {100 * -gap:.1f} points below the trained model ({pct(trained['accuracy'])})."
                       + paired + " The interval is entirely below zero, so the gap is real. "
                       "Inspectors file similar problems under "
                       f"specific codes by habit, and a model trained on {info['n_train']:,} of their notes learns "
                       "those habits; the LLM only sees the category names and official titles.")
        elif gap_low > 0:
            verdict = (f"**The LLM wins.** {best['method']} got {pct(best['accuracy'])} with no training examples, "
                       f"{100 * gap:.1f} points above the trained model ({pct(trained['accuracy'])})." + paired)
        else:
            verdict = (f"**The LLM and the trained model are close.** {best['method']} got {pct(best['accuracy'])} "
                       f"with no training examples, against {pct(trained['accuracy'])} for the trained model."
                       + paired + " The interval includes zero, so this test cannot separate them.")
        verdict += (f" Both are far ahead of the keyword rules ({pct(keywords['accuracy'])}), which only matched "
                    f"anything in {pct(info['keyword_coverage'])} of notes.")

    usage = ""
    if info.get("llm_usage"):
        usage = "\n" + "\n".join(f"- **{m}:** {u}" for m, u in info["llm_usage"].items()) + "\n"

    md = f"""# Can an LLM classify inspection notes?

*Generated by `src/llm_eval.py`. Re-run it to refresh.*

Each violation in Chicago's data has an official code and the inspector's free-text note. The
codes are grouped into {len(groups)} categories ([`data/reference/violation_groups.csv`](../data/reference/violation_groups.csv)),
and every method sees only the note and has to pick the category. The official code is the answer
key, so no hand labeling is needed.

- **Test set:** {len(test)} notes from {test['inspection_date'].min():%b %Y} to {test['inspection_date'].max():%b %Y},
  {PER_GROUP} per category, no duplicates ([`test_notes.csv`](llm_eval/test_notes.csv)).
  With equal counts per category, guessing would score {pct(1 / len(groups))}.
- **Trained model:** {info['n_train']:,} notes from before {TEST_START:%B %Y}. Notes identical to a test note
  were removed so the model cannot simply memorize copy-pasted text.
- **LLM:** zero-shot. It gets the category names and the official violation titles in each one,
  but no labeled examples. Notes are sent {info['batch_size']} at a time with temperature 0.
- **Left out:** {pct(info['excluded_share'])} of notes, under rare or administrative codes (for example
  "previous violation corrected").

## Results

{chr(10).join(table)}

{verdict}

**Do ordinance numbers give the answer away?** Notes often cite the city code broken
(e.g. "7-38-005"). With every digit removed, the trained model scores {pct(trained_no_num['accuracy'])}
(vs. {pct(trained['accuracy'])} with them), so {'its accuracy comes from the words themselves, not the numbers' if trained['accuracy'] - trained_no_num['accuracy'] < 0.05 else 'part of its accuracy comes from these numbers'}.

![results](figures/12_llm_eval.png)

## F1 by category

{chr(10).join(f1_table)}
{usage}
## Limits

- The answer key is the inspector's choice of code. Some notes fit two categories (dirty floors vs.
  dirty equipment), so no method can reach 100%.
- {len(test)} notes give intervals of about ±3 to ±6 points. Small differences between methods are noise.
- The LLM was tested on one prompt with no examples. Adding a few labeled examples to the prompt
  would likely help it.
- This tests classification only. Running the LLM on all 370,000 notes to feed the risk model
  was out of reach on a free tier.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    return path


def make_figure(scores: pd.DataFrame, path: Path = FIGURE_PATH) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    colors = {"keyword": "#a0aec0", "trained": "#2b6cb0", "llm": "#c05621"}
    rows = scores.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.barh(y, rows["accuracy"], color=[colors[k] for k in rows["kind"]], height=0.6)
    ax.errorbar(rows["accuracy"], y, xerr=[rows["accuracy"] - rows["accuracy_low"],
                                           rows["accuracy_high"] - rows["accuracy"]],
                fmt="none", ecolor="#2d3748", capsize=3, lw=1)
    for i, v in enumerate(rows["accuracy"]):
        ax.text(0.015, i, f"{v:.0%}", va="center", color="white", fontsize=9, fontweight="bold")
    chance = 1 / len([c for c in scores.columns if c.startswith("f1_")])
    ax.axvline(chance, color="#718096", ls="--", lw=1)
    ax.text(chance + 0.01, -0.55, "guessing", color="#718096", fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["short"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"Accuracy on {int(scores['n'].iloc[0])} inspection notes (95% interval)")
    ax.set_title("Which method can tell what an inspector's note is about?", fontweight="bold", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- run --

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append",
                        help=f"model to test, as OmniRoute names it (default {DEFAULT_MODEL}); repeat to test several")
    parser.add_argument("--skip-llm", action="store_true", help="do not call the LLM; use saved answers only")
    parser.add_argument("--batch-size", type=int, default=10, help="notes per request (default 10)")
    parser.add_argument("--wait", type=float, default=7.0,
                        help="seconds between requests, to stay under free-tier limits (default 7)")
    parser.add_argument("--timeout", type=float, default=600,
                        help="seconds to wait for one answer before retrying (default 600)")
    parser.add_argument("--base-url", default=os.environ.get("OMNIROUTE_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()
    models = args.model or [DEFAULT_MODEL]

    if not CLEAN_PATH.exists():
        raise SystemExit(f"{CLEAN_PATH} not found. Run src/download_data.py and src/clean.py first.")
    groups = load_groups()
    clean = pd.read_csv(CLEAN_PATH, usecols=["inspection_id", "inspection_date", "violations", "new_food_code"],
                        low_memory=False)
    notes = split_notes(clean)
    labeled = label_notes(notes, groups)
    long_notes = notes[notes["note"].str.len() >= MIN_CHARS]
    excluded_share = 1 - len(labeled) / len(long_notes)

    # The test set is created once and then reused, so saved LLM answers stay valid.
    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH, parse_dates=["inspection_date"])
        print(f"Using the saved test set ({len(test)} notes)")
    else:
        test = make_test_set(labeled)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        test.to_csv(TEST_PATH, index=False)
        print(f"Created the test set ({len(test)} notes)")
    train = training_notes(labeled, test)
    print(f"Training notes: {len(train):,}")

    group_names = sorted(groups["group"].unique())
    results = [("Keyword rules (the risk model's word lists)", "Keyword rules", "keyword",
                keyword_predict(test["note"]))]
    print("Training the TF-IDF model...")
    results.append(("Trained model (TF-IDF + logistic regression)", "Trained model", "trained",
                    train_model(train).predict(test["note"])))
    results.append(("Trained model, numbers removed from notes", "Trained, no numbers", "trained",
                    train_model(train, numbers=False).predict(test["note"].map(strip_numbers))))

    usage = {}
    if not args.skip_llm:
        key = os.environ.get("OMNIROUTE_API_KEY")
        if not key:
            raise SystemExit("OMNIROUTE_API_KEY is not set. Run: export OMNIROUTE_API_KEY=$(cat ~/.omniroute_key)\n"
                             "Or add --skip-llm to build the report from saved answers only.")
        for model in models:
            client = OmniRouteClient(model, args.base_url, key, args.wait, args.timeout)
            try:
                run_llm(test, groups, client, model, args.batch_size)
            except StopRun as err:
                print(f"\nStopped: {err}\n")
            if client.requests:
                usage[model] = (f"{client.requests} requests, {client.tokens['prompt']:,} prompt tokens, "
                                f"{client.tokens['completion']:,} answer tokens in this run")

    for path in sorted(OUT_DIR.glob("predictions_*.jsonl")):
        answers = load_predictions(path)
        model = json.loads(path.read_text().splitlines()[0])["model"] if answers else path.stem
        if len(answers) < len(test):
            print(f"  {model}: {len(answers)}/{len(test)} notes answered; left out of the report until finished")
            continue
        results.append((f"LLM, zero-shot ({model})", model.split("/")[-1], "llm",
                        test["note_id"].map(answers).tolist()))

    trained_pred = results[1][3]
    rows = []
    for m, s, k, p in results:
        d, lo, hi = paired_difference(test["group"], p, trained_pred) if k == "llm" else (np.nan,) * 3
        rows.append({"method": m, "short": s, "kind": k, **score(test["group"], p, group_names),
                     "diff_vs_trained": d, "diff_low": lo, "diff_high": hi})
    scores = pd.DataFrame(rows)
    per_note = test[["note_id", "inspection_id", "code", "group", "note"]].copy()
    for _, s, _, p in results:
        per_note[s] = list(p)
    per_note.to_csv(NOTE_PREDICTIONS_PATH, index=False)
    scores.to_csv(SCORES_PATH, index=False)
    info = {"n_train": len(train), "batch_size": args.batch_size, "excluded_share": excluded_share,
            "keyword_coverage": keyword_coverage(test["note"]), "llm_usage": usage}
    write_report(scores, test, info)
    make_figure(scores)
    print()
    print(scores[["method", "accuracy", "accuracy_low", "accuracy_high", "macro_f1"]].round(3).to_string(index=False))
    print(f"\nReport: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
