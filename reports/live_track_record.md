# Live Track Record

The model in `models/gradient_boosting.joblib` is frozen. Every week it scores each active restaurant, and the scores are committed to this repo before any of those restaurants are inspected. When an inspection result shows up later, it is compared to the score saved earlier.

- **Prediction runs:** 3 (Nov 12, 2026 to Nov 23, 2026)
- **Restaurants scored in the latest run:** 9,121
- **Inspections checked against an earlier prediction:** 38 (8 failed)

**Early numbers.** With fewer than 200 inspections or 30 failures checked, these results can still swing a lot from week to week.

## Did higher-risk restaurants fail more often?

| Risk tier when predicted | Inspections | Failed | Failure rate |
|---|---|---|---|
| High | 9 | 3 | 33.3% |
| Medium | 10 | 1 | 10.0% |
| Low | 19 | 4 | 21.1% |

High = top 20% of scores in that week's run, medium = next 30%, low = bottom 50%.

## Model vs. the simple rule

The simple rule flags a restaurant if it failed its last routine inspection. It flagged **6** of the checked inspections and caught **3** failures. The model's top **6** scores caught **3**.

- **ROC AUC:** 0.567
- **Average days between prediction and inspection:** 2.4

## How to check this yourself

Each file in `live/snapshots/` is named for the day it was made. The git history shows when it was committed. `live/outcomes.csv` lists every checked inspection with its date, its result, and the snapshot its score came from. The snapshot date is always before the inspection date.
