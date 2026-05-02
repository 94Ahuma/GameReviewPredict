import subprocess
import sys

# install langdetect
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Installing langdetect...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect", "-q"])
    from langdetect import detect, LangDetectException

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# everything requirement
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

# Initialize cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()


def is_english(text):
    """Detect text is in English, or discard"""
    try:
        return detect(str(text)) == 'en'
    except LangDetectException:
        return False


def advanced_clean_process(df, min_len=30):
    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str)

    #Language filtering, higher recognition rate
    before = len(df)
    df = df[df["review"].apply(is_english)].copy()
    print(f"Language filtering: {before} -> {len(df)} lines（remove {before - len(df)} lines not in English）")

    cleaned_reviews = []
    vader_scores = []

    for text in df["review"]:
        #use VADER extract emotional scores
        score = sid.polarity_scores(text)['compound']
        vader_scores.append(score)
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        #Lemmatization
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

        cleaned_reviews.append(" ".join(tokens))

    df["cleaned_review"] = cleaned_reviews
    df["vader_score"] = vader_scores

    df = df[df["cleaned_review"].str.len() >= min_len]
    df = df.drop_duplicates(subset=["cleaned_review"]).copy()

    #Label
    df["label"] = df["voted_up"].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
    return df

if __name__ == "__main__":
    print("Loading data...")
    bg3 = pd.read_csv("../data/raw/BG3_reviews_updated.csv")
    animal = pd.read_csv("../data/raw/animal_crossing_reviews.csv")
    stardew = pd.read_csv("../data/raw/stardew_valley_reviews.csv")
    divinity = pd.read_csv("../data/raw/Divinity_Original_Sin_2_reviews.csv")
    print("Processing BG3 with advanced NLP...")
    bg3_clean = advanced_clean_process(bg3)

    print("Processing Animal Crossing with advanced NLP...")
    animal_clean = advanced_clean_process(animal)

    print("Processing Stardew Valley with advanced NLP...")
    stardew_clean = advanced_clean_process(stardew)

    print("Processing Divinity Original Sin 2 with advanced NLP...")
    divinity_clean = advanced_clean_process(divinity)

    #Data balancing
    bg3_sample = pd.concat([
        bg3_clean[bg3_clean["label"] == 1].sample(4500, random_state=42),
        bg3_clean[bg3_clean["label"] == 0]
    ])

    #Save in local
    bg3_sample.to_csv("../data/processed/BG3_reviews_clean_v2.csv", index=False)
    animal_clean.to_csv("../data/processed/animal_crossing_reviews_clean_v2.csv", index=False)
    stardew_clean.to_csv("../data/processed/stardew_valley_reviews_clean_v2.csv", index=False)
    divinity_clean.to_csv("../data/processed/Divinity_Original_Sin_2_reviews_clean_v2.csv", index=False)
    print("Files saved to processed folder.")