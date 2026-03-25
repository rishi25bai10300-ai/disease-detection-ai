import warnings
warnings.filterwarnings("ignore")

import pickle
import time
import pandas as pd

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model        = saved["model"]
all_symptoms = saved["symptoms"]
le           = saved["label_encoder"]

severity_df = pd.read_csv("archive/Symptom-severity.csv")
desc_df     = pd.read_csv("archive/symptom_Description.csv")
prec_df     = pd.read_csv("archive/symptom_precaution.csv")

severity_map = dict(zip(severity_df["Symptom"].str.strip(), severity_df["weight"]))
desc_map     = dict(zip(desc_df["Disease"], desc_df["Description"]))
prec_map     = {
    row["Disease"]: [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    for _, row in prec_df.iterrows()
}

def slow_print(text, delay=0.018):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def get_severity(symptoms):
    score = sum(severity_map.get(s, 0) for s in symptoms)
    if score <= 13:   return "Mild", score
    elif score <= 18: return "Moderate", score
    else:             return "Severe", score

def predict_top3(symptoms):
    vec    = {s: int(s in symptoms) for s in all_symptoms}
    probas = model.predict_proba(pd.DataFrame([vec]))[0]
    top3   = sorted(zip(le.classes_, probas), key=lambda x: -x[1])[:3]
    return top3

def fuzzy_match(sym):
    sym_clean = sym.replace(" ", "_")
    matches = [s for s in all_symptoms if sym_clean in s or s in sym_clean]
    if not matches:
        matches = [s for s in all_symptoms if sym_clean[:4] in s]
    return matches[:4]

def chat():
    while True:
        print("\n" + "="*54)
        slow_print("      AI Disease Detection Assistant")
        print("="*54 + "\n")
        slow_print("Hello! Describe your symptoms and I'll help identify")
        slow_print("possible conditions. Type 'done' when finished.\n")

        entered = []

        while True:
            prompt = "First symptom: " if not entered else f"Symptom {len(entered)+1} (or 'done'): "
            sym = input(prompt).strip().lower().replace(" ", "_")

            if sym in ("done", "no", "n", ""):
                if not entered:
                    print("  Please enter at least one symptom.\n")
                    continue
                break

            if sym in all_symptoms:
                if sym in entered:
                    print(f"  Already added '{sym.replace('_',' ')}'.")
                    continue
                entered.append(sym)
                weight = severity_map.get(sym, 0)
                label  = "mild" if weight <= 2 else "moderate" if weight <= 5 else "severe"
                print(f"  ✓ {sym.replace('_',' ')} [{label}, {weight}/7]")
            else:
                close = fuzzy_match(sym)
                print(f"  I don't recognize '{sym.replace('_',' ')}'.")
                if close:
                    print(f"  Did you mean: {', '.join(s.replace('_',' ') for s in close)}?")

        print("\n  Analyzing", end="", flush=True)
        for _ in range(5):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print("\n")

        top3            = predict_top3(entered)
        disease, conf   = top3[0]
        level, score    = get_severity(entered)
        avg_weight      = sum(severity_map.get(s, 0) for s in entered) / len(entered)

        print("="*54)
        print(f"  Predicted Disease : {disease}")
        print(f"  Confidence        : {conf:.0%}")
        print(f"  Severity          : {level}  (score: {score})")
        print(f"  Symptoms entered  : {len(entered)}")
        print("="*54)

        if len(top3) > 1 and top3[1][1] > 0.1:
            print(f"\n  Other possibilities:")
            for d, p in top3[1:]:
                if p > 0.05:
                    print(f"    - {d} ({p:.0%})")

        if avg_weight <= 2 or conf < 0.6:
            print()
            if avg_weight <= 2:
                slow_print(f"  Your symptoms ({', '.join(s.replace('_',' ') for s in entered)})")
                slow_print("  are common and shared by many conditions.")
            if conf < 0.6:
                slow_print(f"  Confidence is low ({conf:.0%}) — result may not be accurate.")
            slow_print("  Add more specific symptoms for a better prediction.")
            slow_print("  Please consult a doctor for a proper diagnosis.")

        print(f"\n  About {disease}:")
        slow_print(f"  {desc_map.get(disease, 'No description available.')}", delay=0.012)

        precs = prec_map.get(disease, [])
        if precs:
            print("\n  Recommended Precautions:")
            for i, p in enumerate(precs, 1):
                time.sleep(0.15)
                print(f"    {i}. {p}")

        print()
        slow_print("  This tool is for informational purposes only.")
        slow_print("  Always consult a qualified doctor for medical advice.")

        print()
        again = input("  Check another condition? (yes/no) ").strip().lower()
        if again not in ("yes", "y"):
            print()
            slow_print("  Take care. Goodbye!")
            print()
            break

chat()
