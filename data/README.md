# data/

Datasets live here. **They are gitignored** — never committed (both are large and public).

Fetch them with:

```bash
python scripts/fetch_data.py --dataset european   # creditcard.csv (replication target)
python scripts/fetch_data.py --dataset paysim     # paysim.csv     (generalization test)
```

| File | Dataset | Rows | Fraud | Source |
|------|---------|------|-------|--------|
| `creditcard.csv` | European Credit Card Fraud | 284,807 | 492 (0.172%) | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `paysim.csv` | PaySim synthetic mobile money | 1,048,575 | 1,142 (0.109%) | [Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |

> PaySim's label column is `isFraud` (not `Class`) and its features differ — it needs a
> config override and extra preprocessing. That wiring lands in Phase 2 (generalization test).
