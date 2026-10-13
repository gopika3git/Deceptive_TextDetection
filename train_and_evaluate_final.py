"""
Final Model Training and Comprehensive Evaluation Pipeline
Trains the optimal Combined Word + Character TF-IDF Linear SVM on clean leak-free data.
Evaluates the final model on both the untouched test set and the strictly deduplicated test set.
Saves model artifacts, vectorizers, metadata, and evaluation reports.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
FINAL_C = 0.15

print("=" * 80)
print("TRAINING FINAL LEAK-FREE OPTIMIZED MODEL")
print("=" * 80)

# 1. Load data
train_df = pd.read_csv('train_original.csv')
test_df = pd.read_csv('test_original_untouched.csv')

print(f"Original Train shape: {train_df.shape}")
print(f"Original Test shape : {test_df.shape}")

# Deduplication and Leakage Removal
# Identify exact text overlap between train and test
test_clean_set = set(test_df['text_clean'])
train_overlap_mask = train_df['text_clean'].isin(test_clean_set)
overlap_count = train_overlap_mask.sum()
print(f"Leaking reviews detected in training set: {overlap_count}")

# Derived clean training set (persisted separately)
train_leakfree = train_df[~train_overlap_mask].copy().reset_index(drop=True)
train_leakfree.to_csv('train_leakfree.csv', index=False)
print(f"Saved leak-free training dataset: train_leakfree.csv ({len(train_leakfree)} reviews)")

# Derived deduplicated test set (persisted separately)
test_dedup = test_df.drop_duplicates(subset=['text_clean']).copy().reset_index(drop=True)
test_dedup.to_csv('test_deduplicated.csv', index=False)
print(f"Saved deduplicated test dataset: test_deduplicated.csv ({len(test_dedup)} reviews)")

# 2. Vectorizers
print("\nFitting Word TF-IDF Vectorizer...")
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)
X_word_train = word_vectorizer.fit_transform(train_leakfree["text_clean"])
print(f"Word feature shape: {X_word_train.shape}")

print("\nFitting Character TF-IDF Vectorizer...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=40000
)
X_char_train = char_vectorizer.fit_transform(train_leakfree["text_clean"])
print(f"Character feature shape: {X_char_train.shape}")

# 3. Combined Features
X_train = hstack([X_word_train, X_char_train]).tocsr()
y_train = train_leakfree["label"].values
print(f"\nCombined train feature matrix: {X_train.shape}")

# 4. Train Linear SVM
print(f"\nFitting Linear SVM (C={FINAL_C}, random_state={RANDOM_STATE})...")
model = LinearSVC(C=FINAL_C, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
print("Training complete.")

# 5. Save Artifacts
joblib.dump(model, "final_svm_model.pkl")
joblib.dump(word_vectorizer, "final_word_vectorizer.pkl")
joblib.dump(char_vectorizer, "final_char_vectorizer.pkl")
print("\nSaved artifacts: final_svm_model.pkl, final_word_vectorizer.pkl, final_char_vectorizer.pkl")

# 6. Evaluation Function
def evaluate_dataset(name, df_eval):
    X_w = word_vectorizer.transform(df_eval["text_clean"])
    X_c = char_vectorizer.transform(df_eval["text_clean"])
    X_eval = hstack([X_w, X_c]).tocsr()
    y_eval = df_eval["label"].values
    
    preds = model.predict(X_eval)
    decision_scores = model.decision_function(X_eval)
    
    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds, zero_division=0)
    f1 = f1_score(y_eval, preds, zero_division=0)
    cm = confusion_matrix(y_eval, preds)
    report_dict = classification_report(y_eval, preds, target_names=["Truthful", "Deceptive"], output_dict=True)
    report_text = classification_report(y_eval, preds, target_names=["Truthful", "Deceptive"], digits=4)
    
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS: {name}")
    print("="*70)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score : {f1:.4f} ({f1*100:.2f}%)")
    print("\nConfusion Matrix:")
    print("             Truthful  Deceptive")
    print(f"Truthful     {cm[0,0]:8d}  {cm[0,1]:9d}")
    print(f"Deceptive    {cm[1,0]:8d}  {cm[1,1]:9d}")
    print("\nClassification Report:\n", report_text)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "sample_count": len(df_eval),
        "predictions": preds,
        "scores": decision_scores
    }

# Evaluate on untouched test set
eval_untouched = evaluate_dataset("Untouched Original Test Set (320 reviews)", test_df)

# Evaluate on deduplicated test set
eval_dedup = evaluate_dataset("Strictly Deduplicated Leak-Free Test Set (319 reviews)", test_dedup)

# 7. Save Predictions & Metrics
results_df = pd.DataFrame({
    "actual": test_df["label"].values,
    "predicted": eval_untouched["predictions"],
    "decision_score": eval_untouched["scores"]
})
results_df.to_csv("final_test_predictions.csv", index=False)

metrics_df = pd.DataFrame([{
    "accuracy": eval_untouched["accuracy"],
    "precision": eval_untouched["precision"],
    "recall": eval_untouched["recall"],
    "f1": eval_untouched["f1"]
}])
metrics_df.to_csv("final_test_metrics.csv", index=False)

# 8. Save Comprehensive Metadata & Evaluation Report JSON
import sklearn
metadata = {
    "model_name": "Combined Word + Character TF-IDF Linear SVM",
    "model_type": "LinearSVC",
    "hyperparameters": {
        "C": FINAL_C,
        "random_state": RANDOM_STATE
    },
    "feature_config": {
        "word_vectorizer": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "max_features": 40000,
            "sublinear_tf": True
        },
        "char_vectorizer": {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "max_features": 40000,
            "sublinear_tf": True
        },
        "total_features": int(X_train.shape[1])
    },
    "environment": {
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__
    },
    "dataset_sizes": {
        "train_original": len(train_df),
        "train_leakfree": len(train_leakfree),
        "test_original_untouched": len(test_df),
        "test_deduplicated": len(test_dedup)
    },
    "cross_validation_results": {
        "best_cv_accuracy": 0.9039,
        "best_cv_f1": 0.9037,
        "validation_strategy": "5-Fold Stratified Cross-Validation on Training Data"
    },
    "test_results_untouched": {
        "accuracy": eval_untouched["accuracy"],
        "precision": eval_untouched["precision"],
        "recall": eval_untouched["recall"],
        "f1": eval_untouched["f1"],
        "confusion_matrix": eval_untouched["confusion_matrix"]
    },
    "test_results_deduplicated": {
        "accuracy": eval_dedup["accuracy"],
        "precision": eval_dedup["precision"],
        "recall": eval_dedup["recall"],
        "f1": eval_dedup["f1"],
        "confusion_matrix": eval_dedup["confusion_matrix"]
    },
    "audit_findings": {
        "raw_text_leakage_detected": 1,
        "leaking_review_remedied": True,
        "data_augmentation_evaluated": "Evaluated in CV; rejected due to performance degradation compared to natural reviews."
    }
}

with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model_metadata.json, final_test_metrics.csv, and final_test_predictions.csv successfully.")
