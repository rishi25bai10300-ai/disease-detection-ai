# Disease Detection AI

A machine learning system that predicts diseases based on symptoms. Describe what you're feeling, and it tells you what it might be along with a confidence score, severity level, a plain English description of the condition, and recommended precautions.

This is not a replacement for a doctor. It's a starting point.

---

## What it does

- Accepts symptoms conversationally, one at a time
- Predicts the most likely disease with a confidence score
- Shows the top 3 possible conditions
- Calculates severity based on symptom weights (mild / moderate / severe)
- Describes the predicted disease in plain English
- Lists recommended precautions
- Warns when symptoms are too generic for a reliable prediction
- Suggests consulting a doctor when confidence is low

---

## How it works

The model was trained on a dataset of 41 diseases and 132 symptoms. Each symptom combination is encoded as a binary vector and fed into a Gradient Boosting classifier. Validated using 5 fold cross validation achieves ~99% CV accuracy on the dataset.

---

## Setup

**Requirements**
- Python 3.8+
- pip

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Clone the repo**
```bash
git clone https://github.com/yourusername/disease-detection-ai
cd disease-detection-ai
```

**Download the dataset**

Get the files from [Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) and place them in the `archive/` folder:

```
archive/
├── dataset.csv
├── Symptom-severity.csv
├── symptom_Description.csv
└── symptom_precaution.csv
```

---

## Usage

**Step 1 — Train the model**
```bash
python3 training.py
```
Trains 5 classifiers, picks the best one, saves it to `model.pkl`.

**Step 2 — Evaluate (optional)**
```bash
python3 evaluate.py
```
Tests the saved model against 820 synthetic cases. Saves a confusion matrix to `confusion_matrix.png`.

**Step 3 — Run the bot**
```bash
python3 bot.py
```

---

## Example session

```
====================================================
      AI Disease Detection Assistant
====================================================

Hello! Describe your symptoms and I'll help identify
possible conditions. Type 'done' when finished.

First symptom: itching
  ✓ itching [mild, 1/7]
Symptom 2 (or 'done'): skin_rash
  ✓ skin_rash [mild, 3/7]
Symptom 3 (or 'done'): nodal_skin_eruptions
  ✓ nodal_skin_eruptions [moderate, 4/7]
Symptom 4 (or 'done'): done

  Analyzing.....

====================================================
  Predicted Disease : Fungal infection
  Confidence        : 100%
  Severity          : Mild  (score: 8)
  Symptoms entered  : 3
====================================================

  About Fungal infection:
  Fungal infection is caused by a fungus that invades the tissue...

  Recommended Precautions:
    1. bath twice
    2. use detol or neem in bathing water
    3. keep infected area dry
    4. use clean cloths
```

---

## Symptom format

Symptoms must match the dataset's naming format lowercase with underscores. The bot suggests close matches if it doesn't recognize what you typed.

| Type this | Not this |
|---|---|
| `high_fever` | `fever` |
| `skin_rash` | `skin rash` |
| `breathlessness` | `shortness of breath` |

---

## Project structure

```
disease-detection-ai/
├── training.py          # Train and save the model
├── evaluate.py          # Evaluate saved model
├── bot.py               # Interactive chatbot
├── plot_overfit.py      # Learning curve analysis
├── model.pkl            # Saved model (generated after training)
├── REPORT.md            # Full project report
└── archive/
    ├── dataset.csv
    ├── Symptom-severity.csv
    ├── symptom_Description.csv
    └── symptom_precaution.csv
```

---

## Limitations

- Covers 41 diseases only
- Symptoms must match the dataset's naming format
- Based on a synthetic dataset, not real patient records
- Not a medical device always consult a qualified doctor

---

## License

MIT
