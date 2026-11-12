# Live Track Record

The model in `models/gradient_boosting.joblib` is frozen. Every week it scores each active restaurant, and the scores are committed to this repo before any of those restaurants are inspected. When an inspection result shows up later, it is compared to the score saved earlier.

- **Prediction runs:** 1 (Nov 12, 2026 to Nov 12, 2026)
- **Restaurants scored in the latest run:** 9,108
- **Inspections checked against an earlier prediction:** 0 (0 failed)

No inspections have been checked yet. Results appear here once restaurants scored in an earlier run get inspected.

## How to check this yourself

Each file in `live/snapshots/` is named for the day it was made. The git history shows when it was committed. `live/outcomes.csv` lists every checked inspection with its date, its result, and the snapshot its score came from. The snapshot date is always before the inspection date.
