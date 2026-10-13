import pandas as pd
import numpy as np
import random
import re
import joblib

from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

# ============================================================
# STEP 12 - FINAL TRAINING
# ============================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("STEP 12 - FINAL MODEL TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL TRAINING DATA
# ------------------------------------------------------------

df = pd.read_csv("train_original.csv")

print("\nOriginal training rows:", len(df))

# ------------------------------------------------------------
# 2. REMOVE EXACT DUPLICATE REVIEWS
# ------------------------------------------------------------

df["text_clean"] = df["text_clean"].fillna("").astype(str)

before = len(df)

df = df.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).reset_index(drop=True)

removed = before - len(df)

print("Duplicate reviews removed:", removed)
print("Clean training rows:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

# ------------------------------------------------------------
# 3. SYNONYM REPLACEMENT
#    Conservative: replace at most ONE word
# ------------------------------------------------------------

def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            replacement = lemma.name().replace("_", " ")

            if (
                replacement.lower() != word.lower()
                and replacement.isalpha()
                and len(replacement) >= 3
                and len(replacement) <= 15
            ):
                synonyms.add(replacement.lower())

    return list(synonyms)


def synonym_replace(text):
    words = text.split()

    candidates = []

    for i, word in enumerate(words):

        clean_word = re.sub(r"[^a-zA-Z]", "", word)

        if len(clean_word) < 4:
            continue

        synonyms = get_synonyms(clean_word)

        if synonyms:
            candidates.append((i, synonyms))

    if not candidates:
        return text

    index, synonyms = random.choice(candidates)

    replacement = random.choice(synonyms)

    original_word = words[index]

    # Preserve capitalization
    if original_word[0].isupper():
        replacement = replacement.capitalize()

    words[index] = replacement

    return " ".join(words)


# ------------------------------------------------------------
# 4. CONTROLLED PARAPHRASING
# ------------------------------------------------------------

PARAPHRASE_RULES = [
    (r"\bvery good\b", "really good"),
    (r"\bvery nice\b", "really nice"),
    (r"\bvery clean\b", "extremely clean"),
    (r"\bvery comfortable\b", "quite comfortable"),
    (r"\bvery helpful\b", "extremely helpful"),
    (r"\bvery friendly\b", "really friendly"),
    (r"\bvery beautiful\b", "really beautiful"),
    (r"\bvery pleasant\b", "really pleasant"),
    (r"\bvery bad\b", "really poor"),
    (r"\bvery poor\b", "quite poor"),
    (r"\breally good\b", "very good"),
    (r"\breally nice\b", "very nice"),
    (r"\breally clean\b", "very clean"),
    (r"\breally comfortable\b", "very comfortable"),
    (r"\breally friendly\b", "very friendly"),
    (r"\breally helpful\b", "very helpful"),
    (r"\bquite good\b", "fairly good"),
    (r"\bquite nice\b", "fairly nice"),
    (r"\bquite clean\b", "fairly clean"),
    (r"\ba lot\b", "quite a lot"),
    (r"\bI loved\b", "I really liked"),
    (r"\bI love\b", "I really like"),
    (r"\bI liked\b", "I enjoyed"),
    (r"\bI did not like\b", "I disliked"),
    (r"\bI didn't like\b", "I disliked"),
    (r"\bwould definitely\b", "would certainly"),
    (r"\bdefinitely recommend\b", "certainly recommend"),
    (r"\bhighly recommend\b", "strongly recommend"),
    (r"\bexcellent service\b", "great service"),
    (r"\bgreat service\b", "excellent service"),
    (r"\bgreat experience\b", "excellent experience"),
    (r"\bexcellent experience\b", "great experience"),
    (r"\bgreat hotel\b", "excellent hotel"),
    (r"\bexcellent hotel\b", "great hotel"),
    (r"\bbeautiful room\b", "lovely room"),
    (r"\blovely room\b", "beautiful room"),
    (r"\bgood location\b", "convenient location"),
    (r"\bconvenient location\b", "good location"),
]


def controlled_paraphrase(text):

    new_text = text

    rules = PARAPHRASE_RULES.copy()
    random.shuffle(rules)

    changed = False

    for pattern, replacement in rules:

        if re.search(pattern, new_text, flags=re.IGNORECASE):

            new_text = re.sub(
                pattern,
                replacement,
                new_text,
                count=1,
                flags=re.IGNORECASE
            )

            changed = True
            break

    return new_text if changed else None


# ------------------------------------------------------------
# 5. CREATE AUGMENTED TRAINING DATA
# ------------------------------------------------------------

print("\nCreating augmentation...")

original_rows = []

for _, row in df.iterrows():

    original_rows.append({
        "text_clean": row["text_clean"],
        "label": row["label"]
    })


# Synonym augmentation
synonym_rows = []

for _, row in df.iterrows():

    augmented_text = synonym_replace(row["text_clean"])

    if augmented_text != row["text_clean"]:

        synonym_rows.append({
            "text_clean": augmented_text,
            "label": row["label"]
        })


# Controlled paraphrase augmentation
paraphrase_rows = []

for _, row in df.iterrows():

    augmented_text = controlled_paraphrase(row["text_clean"])

    if augmented_text is not None:
        paraphrase_rows.append({
            "text_clean": augmented_text,
            "label": row["label"]
        })


original_df = pd.DataFrame(original_rows)
synonym_df = pd.DataFrame(synonym_rows)
paraphrase_df = pd.DataFrame(paraphrase_rows)

final_train = pd.concat(
    [
        original_df,
        synonym_df,
        paraphrase_df
    ],
    ignore_index=True
)

# Remove accidental duplicate generated reviews
final_train = final_train.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).reset_index(drop=True)


print("\nAugmentation summary:")
print("Original:", len(original_df))
print("Synonym:", len(synonym_df))
print("Paraphrase:", len(paraphrase_df))
print("Final training:", len(final_train))

print("\nFinal class distribution:")
print(final_train["label"].value_counts())

# ------------------------------------------------------------
# 6. SAVE FINAL TRAINING DATA
# ------------------------------------------------------------

final_train.to_csv(
    "final_augmented_training.csv",
    index=False
)

print("\nSaved:")
print("final_augmented_training.csv")

# ------------------------------------------------------------
# 7. TF-IDF - WORD FEATURES
# ------------------------------------------------------------

print("\nCreating WORD TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)

X_word = word_vectorizer.fit_transform(
    final_train["text_clean"]
)

print("Word feature shape:", X_word.shape)

# ------------------------------------------------------------
# 8. TF-IDF - CHARACTER FEATURES
# ------------------------------------------------------------

print("\nCreating CHARACTER TF-IDF features...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)

X_char = char_vectorizer.fit_transform(
    final_train["text_clean"]
)

print("Character feature shape:", X_char.shape)

# ------------------------------------------------------------
# 9. COMBINE WORD + CHARACTER FEATURES
# ------------------------------------------------------------

X_train = hstack(
    [X_word, X_char]
).tocsr()

y_train = final_train["label"].values

print("\nCombined feature shape:", X_train.shape)

# ------------------------------------------------------------
# 10. TRAIN FINAL LINEAR SVM
# ------------------------------------------------------------

print("\nTraining final Linear SVM...")

FINAL_C = 0.15

model = LinearSVC(
    C=FINAL_C,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

print("Training completed.")
print("Final C:", FINAL_C)

# ------------------------------------------------------------
# 11. SAVE EVERYTHING NEEDED FOR FINAL TEST
# ------------------------------------------------------------

joblib.dump(
    model,
    "final_svm_model.pkl"
)

joblib.dump(
    word_vectorizer,
    "final_word_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "final_char_vectorizer.pkl"
)

print("\nSaved final model files:")
print("final_svm_model.pkl")
print("final_word_vectorizer.pkl")
print("final_char_vectorizer.pkl")

# ------------------------------------------------------------
# 12. TEST STATUS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST STATUS")
print("=" * 70)

print("The 320-review test set was NOT loaded.")
print("The test set remains completely untouched.")
print("Final model is ready for Step 13.")
print("=" * 70)