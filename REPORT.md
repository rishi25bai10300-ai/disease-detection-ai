# Disease Detection AI — Project Report

## 1. Problem Statement

When people feel unwell, their first instinct is to search their symptoms online. The results are usually overwhelming a mix of rare conditions, worst case scenarios, and generic advice that doesn't actually help. There's a real gap between "I don't feel well" and "I have a reasonable idea of what this might be."

This project attempts to fill that gap. The goal was to build a symptom based disease prediction system that gives users a structured, confidence-scored result not a diagnosis, but a meaningful starting point before they decide whether to see a doctor.

---

## 2. Why This Problem

- Most people self diagnose online before visiting a doctor
- Generic search results cause unnecessary anxiety or false reassurance
- Early awareness of a condition can lead to faster, better treatment
- In areas with limited healthcare access, even a basic tool can make a difference

This is a learning project, but the underlying problem is real and worth solving properly.

---

## 3. Dataset

**Source:** [Kaggle — Disease Symptom Prediction Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

| Property | Value |
|---|---|
| Diseases | 41 |
| Unique symptoms | 132 |
| Total rows | 4920 |
| Rows per disease | 120 |

**Issues found during exploration:**

- **4616 duplicate rows** — only 304 unique symptom combinations existed. Training and testing on duplicated data produces artificially perfect accuracy. Fixed by deduplicating before splitting.
- **Leading whitespace** in symptom values — `" skin_rash"` and `"skin_rash"` were being treated as different features. Fixed with `.str.strip()`.

These two issues, if left unfixed, would have made every model appear to achieve 100% accuracy which is exactly what happened before the fix.

---

## 4. Methodology

### 4.1 Data Preparation
1. Load CSV and strip whitespace from all symptom columns
2. Remove duplicate rows (4920 → 304 unique)
3. Encode symptoms as a binary feature matrix (one column per symptom, 1 = present, 0 = absent)
4. Encode disease labels as integers using `LabelEncoder`
5. Stratified 75/25 train/test split

### 4.2 Models Trained
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Gradient Boosting | 100% | 98.68% |
| Random Forest | 98.68% | 98.68% |
| KNN | 99.34% | 100% |
| XGBoost | 88.48% | 89.47% |
| Decision Tree | 34.22% | 38.16% |

### 4.3 Evaluation Strategy
- **5-fold stratified cross-validation** for reliable accuracy estimates
- **820 augmented synthetic test cases** (20 per disease) to simulate real world input where users enter different symptom combinations than those in the training data

---

## 5. Key Decisions

**Deduplication before splitting**
Without this, identical rows appear in both train and test sets. The model memorizes rather than learns. Decision Tree accuracy dropped from 100% to 38% after the fix the clearest evidence that the original results were meaningless.

**Cross-validation over a single split**
With only 304 unique rows and 41 diseases (~7 per disease), a single split gives 1–2 test samples per disease. One wrong prediction skews the entire result. CV across 5 folds gives a far more trustworthy estimate.

**Gradient Boosting as the final model**
Consistent 100% CV accuracy with 98.68% test accuracy. Random Forest was close, but Gradient Boosting edged it out. XGBoost underperformed on this small dataset its aggressive boosting leads to overfitting when training data is limited.

**No neural network**
An MLP was tested. With 304 rows, neural networks perform worse than tree based models they need significantly more data to generalize. The right tool for structured tabular data is an ensemble method, not a deep learning architecture.

**Rejecting the IEEE dataset**
A larger dataset (391 diseases, 1325 symptoms) was explored. The problem: only 1 training row per disease. Best achievable accuracy was 61% with KNN. More diseases with less data per disease is strictly worse than fewer diseases with more data per disease.

---

## 6. The Bot

The final interface (`bot.py`) is a conversational terminal application:

- Accepts symptoms one by one with natural prompts
- Validates input against the known symptom list
- Suggests close matches for unrecognized input (fuzzy matching)
- Prevents duplicate symptom entries
- Predicts the top 3 most likely diseases with confidence scores
- Calculates severity from symptom weights (1–7 scale)
- Displays disease description and recommended precautions
- Issues a warning when symptoms are too generic (avg weight ≤ 2) or confidence is below 60%
- Recommends consulting a doctor in uncertain cases
- Loops to allow checking multiple conditions in one session

---

## 7. Challenges

**The 100% accuracy trap**
Every model showed 100% accuracy initially. Rather than accepting this, the data was investigated. The cause was 4616 duplicate rows creating train/test overlap. This was the most important finding in the project.

**Small dataset**
304 unique rows for 41 diseases is genuinely limited. Learning curve analysis showed the model only stabilizes near the full dataset size there's very little margin. Real-world deployment would require substantially more data.

**Symptom naming mismatch**
Users naturally type "fever" but the dataset requires "high_fever" or "mild_fever". Fuzzy matching helps, but it's a fundamental limitation of the dataset's specificity.

---

## 8. What I Learned

**Data quality matters more than model choice.** The biggest accuracy improvement came from fixing duplicate rows not from tuning hyperparameters. A clean dataset with a simple model beats a messy dataset with a complex one.

**Perfect results are a warning sign.** 100% accuracy across six different models should raise suspicion, not satisfaction. It almost always means data leakage.

**Cross-validation is essential on small datasets.** A single split on 304 rows is statistically unreliable. CV gave results that could actually be trusted.

**Match the model to the data.** Neural networks are powerful but wrong for this problem. Gradient Boosting on 304 rows of binary features outperforms an MLP every time.

**The pipeline is the product.** Training a model is a small part of the work. Data cleaning, honest evaluation, saving artifacts correctly, and building a usable interface are where most of the effort goes — and where most of the value is.

---

## 9. Limitations & Future Work

- Dataset is synthetic real patient records would produce more realistic results
- Only 41 diseases covered
- Symptom input requires exact dataset format a natural language interface would improve usability significantly
- No patient history, age, or demographic data considered
- A web interface (Flask or Streamlit) would make this accessible without a terminal

---

## 10. Project Files

| File | Purpose |
|---|---|
| `training.py` | Data preparation, model training, saves best model to `model.pkl` |
| `evaluate.py` | Loads saved model, runs evaluation, generates confusion matrix |
| `bot.py` | Interactive conversational chatbot |
| `plot_overfit.py` | Learning curve plots for overfitting analysis |
| `model.pkl` | Saved model + label encoder + symptom list |
| `REPORT.md` | This document |
| `archive/` | Dataset CSV files |
