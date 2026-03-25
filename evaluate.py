import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model        = saved["model"]
all_symptoms = saved["symptoms"]
lf           = saved["label_encoder"]
model_name   = saved["model_name"]

df = pd.read_csv("archive/dataset.csv")
symptoms = [c for c in df.columns if c != "Disease"]
df[symptoms] = df[symptoms].apply(lambda col: col.str.strip())
df_unique = df.drop_duplicates(subset=symptoms)

def encode_row(row):
    present = set(row[symptoms].dropna().values)
    return [int(s in present) for s in all_symptoms]

X = pd.DataFrame([encode_row(row) for _, row in df_unique.iterrows()], columns=all_symptoms)
y = lf.transform(df_unique["Disease"])

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=48, stratify=y)

random.seed(42)
extra_rows, extra_labels = [], []
for disease, group in df_unique.groupby("Disease"):
    all_s = list({s for col in symptoms for s in group[col].dropna()})
    for _ in range(20):
        chosen = set(random.sample(all_s, random.randint(3, min(6, len(all_s)))))
        extra_rows.append([int(s in chosen) for s in all_symptoms])
        extra_labels.append(lf.transform([disease])[0])

X_test = pd.DataFrame(extra_rows, columns=all_symptoms)
y_test = np.array(extra_labels)

y_pred = model.predict(X_test)
print(f"Model        : {model_name}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\n{classification_report(y_test, y_pred, target_names=lf.classes_)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Per fold     : {cv_scores.round(4)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=lf.classes_, yticklabels=lf.classes_, cmap="Blues")
plt.title(f"Confusion Matrix — {model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")
