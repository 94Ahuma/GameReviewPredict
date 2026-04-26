import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("../data/processed")
print("Loading data...")
bg3 = pd.read_csv(DATA_DIR / "BG3_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])
animal = pd.read_csv(DATA_DIR / "animal_crossing_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])

#Build model pipeline (LogisticRegression)
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'cleaned_review'),
        ('num', MinMaxScaler(), ['vader_score'])
    ]
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Training on BG3 and predicting on Animal Crossing...")
pipeline.fit(bg3, bg3["label"])
animal["predicted_label"] = pipeline.predict(animal)


#Extract and print comments of specific categories
def print_sample_reviews(df, condition_name, condition_mask, sample_size=3):
    print(f"\n{'=' * 50}")
    print(f"{condition_name}")
    print(f"{'=' * 50}")

    samples = df[condition_mask].sample(n=min(sample_size, len(df[condition_mask])), random_state=42)

    for idx, row in samples.iterrows():
        print(
            f"\n[real label]: {'positive review' if row['label'] == 1 else 'negative review'} | [Predict]: {'positive review' if row['predicted_label'] == 1 else 'negative review'}")
        #limit length
        review_text = row['review']
        if len(review_text) > 300:
            review_text = review_text[:300] + " ..."
        print(f"\"{review_text}\"")

print_sample_reviews(animal, "True Positives",
                     (animal["label"] == 1) & (animal["predicted_label"] == 1))

print_sample_reviews(animal, "True Negatives",
                     (animal["label"] == 0) & (animal["predicted_label"] == 0))

print_sample_reviews(animal, "False Positives",
                     (animal["label"] == 0) & (animal["predicted_label"] == 1))

print_sample_reviews(animal, "False Negatives",
                     (animal["label"] == 1) & (animal["predicted_label"] == 0))