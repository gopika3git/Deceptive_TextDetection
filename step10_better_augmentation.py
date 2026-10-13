import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

from nltk.corpus import wordnet


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

print("=" * 70)
print("STEP 10 - IMPROVED AUGMENTATION")
print("=" * 70)

df = pd.read_csv("train_original.csv")

print("\nOriginal dataset:", df.shape)


# ============================================================
# 2. REMOVE EXACT DUPLICATES BEFORE SPLITTING
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["text_clean"],
    keep="first"
).reset_index(drop=True)

print("Duplicate reviews removed:", before - len(df))
print("Clean dataset:", df.shape)


# ============================================================
# 3. ORIGINAL SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nTraining originals:", len(train_df))
print("Validation originals:", len(val_df))


# ============================================================
# 4. CONSERVATIVE SYNONYM REPLACEMENT
# ============================================================

def get_good_synonym(word):

    synonyms = set()

    for syn in wordnet.synsets(word):

        # Only use the first sense of each word.
        # This reduces strange substitutions.
        for lemma in syn.lemmas():

            candidate = lemma.name().replace("_", " ").lower()

            if (
                candidate != word.lower()
                and candidate.isalpha()
                and len(candidate) >= 3
            ):
                synonyms.add(candidate)

    return list(synonyms)


def conservative_synonym_replace(text):

    words = text.split()

    if len(words) < 8:
        return text

    # Only replace ONE word.
    indices = list(range(len(words)))
    random.shuffle(indices)

    for index in indices:

        original = words[index]

        clean_word = "".join(
            c for c in original
            if c.isalpha()
        )

        if len(clean_word) < 5:
            continue

        synonyms = get_good_synonym(clean_word)

        if not synonyms:
            continue

        # Avoid huge semantic changes.
        candidates = [
            s for s in synonyms
            if abs(len(s) - len(clean_word)) <= 5
        ]

        if not candidates:
            continue

        replacement = random.choice(candidates)

        # Preserve basic capitalization.
        if original[0].isupper():
            replacement = replacement.capitalize()

        words[index] = replacement

        return " ".join(words)

    return text


# ============================================================
# 5. CONTROLLED PARAPHRASING
# ============================================================

PARAPHRASE_RULES = [

    ("very good", "really good"),
    ("really good", "very good"),

    ("very nice", "really nice"),
    ("really nice", "very nice"),

    ("very clean", "really clean"),
    ("really clean", "very clean"),

    ("very comfortable", "really comfortable"),
    ("really comfortable", "very comfortable"),

    ("highly recommend", "strongly recommend"),
    ("strongly recommend", "highly recommend"),

    ("would recommend", "would definitely recommend"),

    ("great hotel", "excellent hotel"),
    ("excellent hotel", "great hotel"),

    ("great experience", "excellent experience"),
    ("excellent experience", "great experience"),

    ("good experience", "pleasant experience"),
    ("pleasant experience", "good experience"),

    ("nice hotel", "pleasant hotel"),
    ("pleasant hotel", "nice hotel"),

    ("great stay", "excellent stay"),
    ("excellent stay", "great stay"),

    ("good stay", "pleasant stay"),
    ("pleasant stay", "good stay"),

    ("friendly staff", "helpful staff"),
    ("helpful staff", "friendly staff"),

    ("clean room", "neat room"),
    ("neat room", "clean room"),

    ("comfortable room", "pleasant room"),
    ("pleasant room", "comfortable room"),

    ("good location", "convenient location"),
    ("convenient location", "good location")
]


def controlled_paraphrase(text):

    lower_text = text.lower()

    possible = []

    for old, new in PARAPHRASE_RULES:

        if old in lower_text:
            possible.append((old, new))

    if not possible:
        return text

    old, new = random.choice(possible)

    position = lower_text.find(old)

    result = (
        text[:position]
        + new
        + text[position + len(old):]
    )

    return result


# ============================================================
# 6. GENERATE CONSERVATIVE AUGMENTATION
# ============================================================

augmented_rows = []

for _, row in train_df.iterrows():

    text = row["text_clean"]
    label = row["label"]

    # Original
    augmented_rows.append({
        "text_clean": text,
        "label": label,
        "augmentation": "original"
    })

    # ONE synonym variant
    synonym_text = conservative_synonym_replace(text)

    if synonym_text != text:

        augmented_rows.append({
            "text_clean": synonym_text,
            "label": label,
            "augmentation": "synonym"
        })

    # ONE controlled paraphrase variant
    paraphrase_text = controlled_paraphrase(text)

    if paraphrase_text != text:

        augmented_rows.append({
            "text_clean": paraphrase_text,
            "label": label,
            "augmentation": "paraphrase"
        })


augmented_df = pd.DataFrame(augmented_rows)

# Remove exact duplicates
augmented_df = augmented_df.drop_duplicates(
    subset=["text_clean", "label"]
).reset_index(drop=True)


print("\n" + "=" * 70)
print("AUGMENTATION RESULTS")
print("=" * 70)

print("Original training:", len(train_df))
print("Augmented training:", len(augmented_df))

print("\nAugmentation distribution:")
print(augmented_df["augmentation"].value_counts())


# ============================================================
# 7. CHECK TRAIN / VALIDATION OVERLAP
# ============================================================

train_texts = set(
    augmented_df["text_clean"]
    .str.lower()
    .str.strip()
)

val_texts = set(
    val_df["text_clean"]
    .str.lower()
    .str.strip()
)

overlap = train_texts.intersection(val_texts)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print("Exact overlap:", len(overlap))

if len(overlap) != 0:
    print("ERROR - overlap detected")
    raise SystemExit

print("PASS - No overlap.")


# ============================================================
# 8. PREPARE TEXT
# ============================================================

X_train_text = augmented_df["text_clean"].fillna("")
X_val_text = val_df["text_clean"].fillna("")

y_train = augmented_df["label"].values
y_val = val_df["label"].values


# ============================================================
# 9. WORD TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("WORD TF-IDF")
print("=" * 70)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)

print("Train:", X_train_word.shape)
print("Validation:", X_val_word.shape)


# ============================================================
# 10. CHARACTER TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("CHARACTER TF-IDF")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)

print("Train:", X_train_char.shape)
print("Validation:", X_val_char.shape)


# ============================================================
# 11. COMBINE
# ============================================================

X_train = hstack([
    X_train_word,
    X_train_char
]).tocsr()

X_val = hstack([
    X_val_word,
    X_val_char
]).tocsr()

print("\nCombined train:", X_train.shape)
print("Combined validation:", X_val.shape)


# ============================================================
# 12. TEST C VALUES
# ============================================================

C_VALUES = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15
]

results = []

best_model = None
best_accuracy = -1
best_f1 = -1
best_C = None


print("\n" + "=" * 70)
print("SVM OPTIMIZATION")
print("=" * 70)


for C in C_VALUES:

    model = LinearSVC(
        C=C,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    print("\nC =", C)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    results.append({
        "C": C,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    if (
        accuracy > best_accuracy
        or (
            accuracy == best_accuracy
            and f1 > best_f1
        )
    ):

        best_accuracy = accuracy
        best_f1 = f1
        best_C = C
        best_model = model


# ============================================================
# 13. RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best C:", best_C)
print("Accuracy :", round(best_accuracy, 4))
print("F1 Score :", round(best_f1, 4))


# ============================================================
# 14. SAVE
# ============================================================

augmented_df.to_csv(
    "step10_augmented_training.csv",
    index=False
)

results_df.to_csv(
    "step10_results.csv",
    index=False
)

with open(
    "step10_best_model.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "model": best_model,
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "C": best_C,
            "validation_accuracy": best_accuracy,
            "validation_f1": best_f1
        },
        f
    )


print("\nSaved:")
print("step10_augmented_training.csv")
print("step10_results.csv")
print("step10_best_model.pkl")

print("\nTEST STATUS:")
print("The 320-review test set was NOT loaded.")
print("The test set remains untouched.")
