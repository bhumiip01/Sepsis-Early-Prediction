# Sepsis Early Prediction — PhysioNet/CinC Challenge 2019

Hour-by-hour sepsis early warning from ICU time-series. A gradient-boosted baseline and an LSTM are trained on the same task, compared on the metric that matters under severe class imbalance, and analyzed for **lead time** (how early the alarm fires) and **alarm fatigue** (the false-alarm cost).

## Problem

Sepsis is time-critical: earlier detection saves lives, but an over-eager alarm floods clinicians with false positives and gets ignored. The task is, at **each hour** of an ICU stay, to predict whether the patient is heading toward sepsis — early enough to act, without crying wolf.

## Dataset

[PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) — 40,336 ICU patients, one row per hour, 40 features (vitals, labs, demographics) plus `SepsisLabel`. For septic patients the label flips to 1 **six hours before** clinical onset.

From the EDA:
- **~6.8%** of patients ever become septic
- only **~1.8%** of all patient-hours are positive — severe class imbalance
- median ICU stay **38 hours**
- labs are **96–99.8% missing**; vitals are far more complete

## Approach

Two models, same per-hour prediction task:

- **Baseline** — gradient-boosted trees (`HistGradientBoostingClassifier`) on each patient-hour independently (no temporal memory). The bar the sequence model must beat.
- **LSTM** — a sequence model over each patient's hourly feature vectors, predicting at every hour, so it can use trend and history.

Preprocessing (standard ICU time-series handling):
- **informative-missingness flags** for key labs (the fact a test was ordered is itself a signal)
- **forward-fill** within each patient
- **patient-level** 70/15/15 split — a patient never appears in two splits (no leakage)
- **train-only** median imputation and standardization

Imbalance is handled with class-weighted training (`pos_weight` / balanced sample weights), and models are judged by **AUPRC**, not accuracy or AUROC (a "never sepsis" model scores ~98% accuracy and is useless).

## Results

### Baseline vs LSTM (test set)

| Model | AUROC | AUPRC |
|---|---|---|
| Baseline (GBDT) | 0.798 | 0.092 |
| **LSTM** | **0.801** | **0.116** |

The LSTM improves **AUPRC ~26%** over the baseline (chance level ≈ 0.018, the positive rate). AUROC is essentially tied — the sequence model's advantage is concentrated in **precision–recall**, i.e. in finding the rare positives, which is exactly what matters here.

### Lead time & alarm fatigue (test: 432 septic, 5,619 non-septic patients)

For each septic patient, how many hours **before clinical onset** the LSTM first alarms, swept across thresholds:

| Alarm on | Patients caught | Median lead | False-alarm rate |
|---|---|---|---|
| top 10% riskiest hours | 65% | 22 h | 22% |
| top 5% | 47% | 20 h | 10% |
| top 1% | 20% | 24 h | 1% |

**When it catches sepsis, it does so ~20–24 hours early** — a clinically meaningful warning window. The median lead is nearly constant across thresholds: the threshold changes *which* and *how many* patients are caught, not *how early*.

## Key findings

- The **LSTM beats the baseline on AUPRC** — the metric that matters under imbalance — while tying on AUROC.
- **~20–24 h median lead time** when a septic patient is caught.
- The central result is the **sensitivity vs. alarm-fatigue tradeoff**: loosen the alarm to catch 65% of cases and ~1 in 4.5 healthy patients is falsely flagged; tighten it to a 1% false-alarm rate and you catch only 20% of cases. You cannot have high sensitivity *and* low false alarms — the core barrier for real-world sepsis early-warning systems.

## Limitations

- **Single run**, modest 1-layer LSTM (64 hidden units, 5 epochs), untuned — the numbers would move with more seeds and tuning.
- "False-alarm rate" here is **per-patient** (ever flagged); the per-hour alarm burden is higher.
- Sequences are **capped at the last 72 hours**; no external validation across the two hospital sources.
- **Not deployable as-is** — this is a research result that quantifies the signal and its cost, not a clinical tool.

## Next steps

- Multi-seed runs + tuning (GRU vs LSTM, depth, class weighting); the challenge's official utility score.
- **Calibration** — are the predicted probabilities trustworthy? (reliability curves)
- Per-hour (not just per-patient) false-alarm analysis.
- **External validation** across hospital A vs. hospital B.

## Reproduce

Open `sepsis_pipeline.ipynb` in Google Colab → **Runtime → Change runtime type → GPU** → **Run all**. The notebook downloads the data from PhysioNet's AWS mirror, runs the full pipeline (EDA → preprocessing → baseline → LSTM → comparison → lead-time), and saves a results figure and CSV. Seeded (`SEED=42`) throughout.

## Stack

Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib
Data: PhysioNet/CinC Challenge 2019 (CC-BY 4.0) — see the [dataset page](https://physionet.org/content/challenge-2019/1.0.0/) for the official citation.
