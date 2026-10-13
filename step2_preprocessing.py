# ============================================================
# DECEPTION DETECTION PROJECT
# STEP 2 - PART 1
# LOAD, CLEAN, SHUFFLE AND SPLIT DATA
# ============================================================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "deceptive-opinion.csv"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ============================================================
# 3. DISPLAY BASIC INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Number of rows   :", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n========================================")
print("FIRST 5 ROWS")
print("========================================")

print(df.head())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(df.isnull().sum())


# ============================================================
# 6. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n========================================")
print("ORIGINAL LABEL DISTRIBUTION")
print("========================================")

print(df["deceptive"].value_counts())


# ============================================================
# 7. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert everything to lowercase
    text = text.lower()

    # Remove line breaks and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at beginning/end
    text = text.strip()

    return text


# ============================================================
# 8. CLEAN REVIEW TEXT
# ============================================================

print("\nCleaning review text...")

df["text_clean"] = df["text"].apply(clean_text)

print("Text cleaning completed!")


# ============================================================
# 9. ENCODE LABEL
# ============================================================

# truthful = 0
# deceptive = 1

label_mapping = {
    "truthful": 0,
    "deceptive": 1
}

df["label"] = (
    df["deceptive"]
    .str.lower()
    .str.strip()
    .map(label_mapping)
)


# ============================================================
# 10. CHECK LABEL ENCODING
# ============================================================

print("\n========================================")
print("LABEL ENCODING")
print("========================================")

print(
    df[["deceptive", "label"]]
    .drop_duplicates()
    .sort_values("label")
)


# ============================================================
# 11. CHECK FOR INVALID LABELS
# ============================================================

if df["label"].isnull().any():

    print("\nERROR: Invalid label detected!")

    print(
        df.loc[
            df["label"].isnull(),
            "deceptive"
        ].unique()
    )

    raise ValueError(
        "Dataset contains labels other than truthful/deceptive."
    )


# ============================================================
# 12. SHUFFLE DATASET
# ============================================================

print("\nShuffling dataset...")

df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("Dataset shuffled successfully!")


# ============================================================
# 13. PREPARE X AND Y
# ============================================================

X = df["text_clean"]
y = df["label"]


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# 15. CREATE TRAINING DATAFRAME
# ============================================================

train_df = df.loc[X_train.index].copy()

test_df = df.loc[X_test.index].copy()


# Reset indexes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 16. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Total dataset :", len(df))
print("Training set  :", len(train_df))
print("Testing set   :", len(test_df))


# ============================================================
# 17. TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

print(
    train_df["deceptive"].value_counts()
)


# ============================================================
# 18. TEST CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("TEST CLASS DISTRIBUTION")
print("========================================")

print(
    test_df["deceptive"].value_counts()
)


# ============================================================
# 19. SHOW CLEANING EXAMPLE
# ============================================================

print("\n========================================")
print("CLEANING EXAMPLE")
print("========================================")

print("\nOriginal:")
print(df["text"].iloc[0])

print("\nCleaned:")
print(df["text_clean"].iloc[0])


# ============================================================
# 20. SAVE TRAIN AND TEST DATA
# ============================================================

train_df.to_csv(
    "train_original.csv",
    index=False
)

test_df.to_csv(
    "test_original_untouched.csv",
    index=False
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("STEP 2 PART 1 COMPLETE")
print("========================================")

print("Created:")
print("1. train_original.csv")
print("2. test_original_untouched.csv")

print("\nIMPORTANT:")
print("The test set will NOT be augmented.")
print("The test set will remain untouched until final evaluation.")