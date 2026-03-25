import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("archive/dataset.csv")
symptoms = [c for c in df.columns if c != "Disease"]
df[symptoms] = df[symptoms].apply(lambda col: col.str.strip())
df_unique = df.drop_duplicates(subset=symptoms)

all_symptoms = sorted({s for col in symptoms for s in df_unique[col].dropna() if s})

def encode_row(row):
    present = set(row[symptoms].dropna().values)
    return [int(s in present) for s in all_symptoms]

X = pd.DataFrame([encode_row(row) for _, row in df_unique.iterrows()], columns=all_symptoms)
lf = LabelEncoder()
y = lf.fit_transform(df_unique["Disease"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=48, stratify=y)

models = {
    "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=48, n_jobs=-1, criterion='entropy', min_samples_split=5, min_samples_leaf=2, max_depth=5, max_features='sqrt', bootstrap=True, class_weight='balanced', oob_score=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=48, learning_rate=0.15, max_depth=5, max_features='sqrt', subsample=0.8, min_samples_split=5, min_samples_leaf=2),
    "XGBoost":           XGBClassifier(n_estimators=300, random_state=48, learning_rate=0.4, max_depth=5, n_jobs=-1, gamma=0.1, eval_metric='mlogloss'),

    "KNN":               KNeighborsClassifier(n_neighbors=8, weights='uniform', metric='minkowski', n_jobs=-1),
    "Decision Tree":     DecisionTreeClassifier(random_state=48, criterion='entropy', min_samples_split=5, min_samples_leaf=2, max_depth=5, max_features='sqrt', class_weight='balanced')
}

best_model, best_name, best_acc = None, "", 0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)

for name, clf in models.items():
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"{name}")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {acc:.2%}\n")
    if cv_scores.mean() > best_acc:
        best_acc, best_name, best_model = cv_scores.mean(), name, clf

print(f"Best: {best_name} ({best_acc:.2%})")

with open("model.pkl", "wb") as f:
    pickle.dump({
        "model":         best_model,
        "symptoms":      all_symptoms,
        "label_encoder": lf,
        "model_name":    best_name
    }, f)

print("Saved to model.pkl")
