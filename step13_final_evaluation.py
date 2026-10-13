import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("STEP 13 - FINAL TEST EVALUATION")
print("=" * 70)

# ============================================================
# 1. LOAD FINAL MODEL
# ============================================================

model = joblib.load("final_svm_model.pkl")

word_vectorizer = joblib.load(
    "final_word_vectorizer.pkl"
)

char_vectorizer = joblib.load(
    "final_char_vectorizer.pkl"
)

print("\nFinal model loaded successfully.")

# ============================================================
# 2. LOAD LOCKED TEST SET
# ============================================================

test = pd.read_csv(
    "test_original_untouched.csv"
)

test["text_clean"] = test["text_clean"].fillna("").astype(str)

print("\nTest set loaded.")
print("Test reviews:", len(test))

print("\nTest class distribution:")
print(test["label"].value_counts())

# ============================================================
# 3. TRANSFORM TEST DATA
# ============================================================

print("\nCreating test TF-IDF features...")

X_word_test = word_vectorizer.transform(
    test["text_clean"]
)

X_char_test = char_vectorizer.transform(
    test["text_clean"]
)

from scipy.sparse import hstack

X_test = hstack(
    [X_word_test, X_char_test]
).tocsr()

y_test = test["label"].values

print("Test feature shape:", X_test.shape)

# ============================================================
# 4. FINAL PREDICTION
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

# ============================================================
# 5. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

# ============================================================
# 6. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted")
print("\n             Truthful  Deceptive")
print(f"Truthful     {cm[0,0]:8d}  {cm[0,1]:9d}")
print(f"Deceptive    {cm[1,0]:8d}  {cm[1,1]:9d}")

# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Truthful", "Deceptive"],
        digits=4
    )
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
})

results.to_csv(
    "final_test_predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}])

metrics.to_csv(
    "final_test_metrics.csv",
    index=False
)

print("\nSaved:")
print("final_test_predictions.csv")
print("final_test_metrics.csv")

# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

print("The final model was evaluated on the untouched 320-review test set.")
print("This is the final unbiased test result.")
print("=" * 70)
