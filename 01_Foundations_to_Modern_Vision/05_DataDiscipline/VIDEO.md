# Episode 05 — Datasets, Leakage, Metrics & Reproducibility

Suggested title:

**Your Model Metric Might Be Lying — Splits, Leakage & Reproducibility**

Suggested thumbnail:

**99% ACCURACY. WRONG MODEL.**

Target duration: 45–55 minutes.

## Chapters

```text
00:00 Why experiment design matters
05:00 Train vs validation vs test
11:00 Exact and near-duplicate leakage
17:00 Video / identity / temporal leakage
23:00 Stable identity-based splitting
30:00 Confusion matrix
35:00 Accuracy vs precision vs recall
41:00 Reproducibility beyond random seeds
46:00 C++ metric/split implementation
50:00 Python ↔ C++ parity
```

## Demonstration

```bash
python 05_DataDiscipline/python/data_discipline_walkthrough.py

cmake -S 05_DataDiscipline -B build/module05
cmake --build build/module05
ctest --test-dir build/module05 --output-on-failure

python 05_DataDiscipline/python/data_discipline_native_parity.py \
  build/module05/cv9x_data_discipline
```

## Core message

A trustworthy ML result is not just a metric. It is a metric attached to a controlled split, provenance, and reproducible evaluation procedure.
