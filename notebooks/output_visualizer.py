import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
#Theme modeling
from bertopic import BERTopic

DATA_DIR = Path("../data/processed")
FIGURES_DIR = Path("../results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#data
print("Loading cleaned data...")
bg3 = pd.read_csv(DATA_DIR / "BG3_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])
animal = pd.read_csv(DATA_DIR / "animal_crossing_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])
stardew  = pd.read_csv(DATA_DIR / "stardew_valley_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])
divinity = pd.read_csv(DATA_DIR / "Divinity_Original_Sin_2_reviews_clean_v2.csv").dropna(subset=["cleaned_review", "review"])

#Steam UI Noise Word Blacklist
STEAM_UI_NOISE = {
    "expand", "click", "view", "contains", "spoiler",
    "hide", "show", "read", "more", "less", "posted",
    "hours", "record", "early", "access", "review"
}

# ==========================================
#Cross game style prediction confusion matrix
# ==========================================
def plot_confusion_matrix_single(pipeline, test_df, test_name, filename):
    y_pred = pipeline.predict(test_df)
    cm = confusion_matrix(test_df["label"], y_pred, labels=[1, 0])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive (Predicted)", "Negative (Predicted)"],
                yticklabels=["Positive (Actual)", "Negative (Actual)"])
    plt.title(f"Confusion Matrix: BG3 Model → {test_name}\n"
              f"(Top-left = TP: correctly predicted positive)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()
    print(f"Saved：{filename}")


def plot_all_confusion_matrices():
    print("Training LR model for confusion matrices...")
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
    pipeline.fit(bg3, bg3["label"])

    plot_confusion_matrix_single(pipeline, animal,   "Animal Crossing (Different Genre)",  "CM_BG3_AnimalCrossing.png")
    plot_confusion_matrix_single(pipeline, divinity, "Divinity Original Sin 2 (Same Genre - RPG)", "CM_BG3_Divinity2.png")
    plot_confusion_matrix_single(pipeline, stardew,  "Stardew Valley (Different Genre)",   "CM_BG3_StardewValley.png")

#BERTopic
def run_fixed_topic_modeling(df, title, filename):
    print(f"\nGenerating theme analysis for {title}...")
    docs = df["cleaned_review"].head(3000).tolist()
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extended_stop_words = list(ENGLISH_STOP_WORDS) + list(STEAM_UI_NOISE)

    vectorizer_model = CountVectorizer(
        stop_words=extended_stop_words,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        min_df=2,
        #Dynamic calculation to avoid max_df < min_df
        max_df=max(3, int(len(docs) * 0.95))
    )

    topic_model = BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        min_topic_size=15
    )

    topics, _ = topic_model.fit_transform(docs)

    freq = topic_model.get_topic_info()
    top_topics = freq[freq['Topic'] != -1].head(5)['Topic'].tolist()

    fig = topic_model.visualize_barchart(topics=top_topics, n_words=10, width=300, height=300)
    fig.write_html(FIGURES_DIR / filename)
    print(f"Saved：{filename}")


#Specific classification example output
def show_sample_classifications():
    print("\n" + "=" * 40)
    print("Extracting specific comment examples for model judgment...")

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(bg3["cleaned_review"])
    svm = LinearSVC().fit(X, bg3["label"])

    sample_animal = animal.head(20).copy()
    X_test = tfidf.transform(sample_animal["cleaned_review"])
    sample_animal["pred"] = svm.predict(X_test)

    for label_type in [1, 0]:
        type_str = "Positive" if label_type == 1 else "Negative"
        print(f"\n--- Example of Model【{type_str}】---")
        subset = sample_animal[sample_animal["pred"] == label_type].head(2)
        for _, row in subset.iterrows():
            print(f"{row['review'][:150]}...")


#VADER Score distribution comparison (pos vs neg, BG3 vs animal)
def plot_vader_distribution():
    print("\nGenerating VADER Score distribution map...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    datasets = [("Baldur's Gate 3", bg3), ("Animal Crossing", animal), ("Divinity Original Sin 2", divinity),
        ("Stardew Valley", stardew),]

    for ax, (title, df) in zip(axes, datasets):
        pos = df[df["label"] == 1]["vader_score"]
        neg = df[df["label"] == 0]["vader_score"]
        ax.hist(neg, bins=40, alpha=0.6, color="#E05C5C", label=f"Negative (n={len(neg)})")
        ax.hist(pos, bins=40, alpha=0.6, color="#5B8FD4", label=f"Positive (n={len(pos)})")
        ax.axvline(pos.mean(), color="#2962A8", linestyle="--",
                   linewidth=1.5, label=f"Pos mean={pos.mean():.2f}")
        ax.axvline(neg.mean(), color="#B02020", linestyle="--",
                   linewidth=1.5, label=f"Neg mean={neg.mean():.2f}")

        ax.set_title(f"{title}: VADER Score Distribution", fontsize=12)
        ax.set_xlabel("VADER Compound Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.suptitle("VADER Sentiment Score: Positive vs Negative Reviews", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "VADER_Distribution.png", dpi=300)
    plt.close()
    print("Saved：VADER_Distribution.png")
    print("\nVADER Score Mean Summary:")
    for title, df in datasets:
        pos_mean = df[df["label"] == 1]["vader_score"].mean()
        neg_mean = df[df["label"] == 0]["vader_score"].mean()
        print(f"  {title} — Positive: {pos_mean:.4f} | Negative: {neg_mean:.4f} "
              f"| Difference: {pos_mean - neg_mean:.4f}")


if __name__ == "__main__":
    plot_all_confusion_matrices()

    run_fixed_topic_modeling(bg3, "Baldur's Gate 3", "BG3_Topics_Fixed.html")
    run_fixed_topic_modeling(animal, "Animal Crossing", "Animal_Topics_Fixed.html")
    run_fixed_topic_modeling(divinity, "Divinity Original Sin 2", "Divinity_Topics_Fixed.html")
    run_fixed_topic_modeling(stardew, "Stardew Valley", "Stardew_Topics_Fixed.html")

    show_sample_classifications()
    plot_vader_distribution()
    print("\nAll visual outputs completed, check the results/figures folder.")