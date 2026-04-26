import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

#Set path
DATA_DIR = Path("../data/processed")
RESULTS_DIR = Path("../results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

bg3 = pd.read_csv(DATA_DIR / "BG3_reviews_clean_v2.csv")
animal = pd.read_csv(DATA_DIR / "animal_crossing_reviews_clean_v2.csv")

def get_pipeline(model_type):
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'cleaned_review'),
            ('num', MinMaxScaler(), ['vader_score'])
        ]
    )
    if model_type == 'NB':
        clf = MultinomialNB()
    elif model_type == 'SVM':
        clf = LinearSVC(C=1.0, max_iter=2000)
    else:
        clf = LogisticRegression(max_iter=1000)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])


def evaluate(y_true, y_pred, model_name, train_set, test_set):
    """Calculate complete evaluation indicators"""
    return {
        "Model":     model_name,
        "Train":     train_set,
        "Test":      test_set,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        #Macro: Calculate separately for each category and then average again
        "F1_macro":  round(f1_score(y_true, y_pred, average='macro'), 4),
        "Precision_macro": round(precision_score(y_true, y_pred, average='macro'), 4),
        "Recall_macro":    round(recall_score(y_true, y_pred, average='macro'), 4),
    }


#training data
train_bg3, test_bg3 = train_test_split(bg3, test_size=0.2, random_state=42)

models_to_test = {
    "LogisticRegression": get_pipeline('LR'),
    "NaiveBayes":         get_pipeline('NB'),
    "LinearSVM":          get_pipeline('SVM')
}

#5-Fold Cross-Validation
print("="*50)
print("5-Fold Cross-Validation on BG3 (stability check)")
print("="*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
cv_results = []

for name, model_type in [("LogisticRegression", 'LR'),
                          ("NaiveBayes", 'NB'),
                          ("LinearSVM", 'SVM')]:
    pipe_cv = get_pipeline(model_type)
    scores = cross_validate(pipe_cv, bg3, bg3["label"],
                            cv=cv, scoring=cv_scoring, n_jobs=1)
    cv_results.append({
        "Model":              name,
        "CV_Accuracy_mean":   round(scores['test_accuracy'].mean(), 4),
        "CV_Accuracy_std":    round(scores['test_accuracy'].std(), 4),
        "CV_F1_macro_mean":   round(scores['test_f1_macro'].mean(), 4),
        "CV_F1_macro_std":    round(scores['test_f1_macro'].std(), 4),
    })
    print(f"{name}: Accuracy {scores['test_accuracy'].mean():.4f} ± "
          f"{scores['test_accuracy'].std():.4f} | "
          f"F1_macro {scores['test_f1_macro'].mean():.4f} ± "
          f"{scores['test_f1_macro'].std():.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
print(f"CV result has been saved to {RESULTS_DIR / 'cv_results.csv'}\n")

results = []

#Cross game style experimental cycle
for name, pipe in models_to_test.items():
    print(f"\n{'='*50}")
    print(f"Running {name}...")

    pipe.fit(train_bg3, train_bg3["label"])

    #test in game style(BG3 -> BG3)
    preds_self = pipe.predict(test_bg3)
    results.append(evaluate(test_bg3["label"], preds_self, name, "BG3", "BG3"))

    print(f"\n[{name}] BG3 → BG3 Classification Report:")
    print(classification_report(test_bg3["label"], preds_self,
                                target_names=["Negative", "Positive"]))

    #test in different game style(BG3 -> Animal Crossing)
    preds_cross = pipe.predict(animal)
    results.append(evaluate(animal["label"], preds_cross, name, "BG3", "AnimalCrossing"))

    print(f"[{name}] BG3 → Animal Crossing Classification Report:")
    print(classification_report(animal["label"], preds_cross,
                                target_names=["Negative", "Positive"]))

#ouput
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("Final Results:")
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_DIR / "advanced_model_results.csv", index=False)
print(f"\n result has been saved to {RESULTS_DIR / 'advanced_model_results.csv'}")