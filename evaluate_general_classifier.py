"""
evaluate_general_classifier.py
Trains and evaluates candidate general-domain deception detection models on the standardized leak-free splits.
Compares:
1. Word-level TF-IDF + Linear SVM
2. Char-level TF-IDF + Linear SVM
3. Combined Word+Char TF-IDF + Linear SVM (Baseline)
4. Combined Word+Char TF-IDF + Calibrated Linear SVM (Platt Scaling)
Outputs verified metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, AUC) to general_model_metrics.json.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    import re
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def run_evaluation():
    print("=== EVALUATING GENERAL CLASSIFIER CANDIDATES ===")
    
    # Load standardized splits
    train_df = pd.read_csv("split_train.csv")
    val_df = pd.read_csv("split_val.csv")
    test_df = pd.read_csv("split_test.csv")
    
    X_train_raw = train_df["text"].apply(clean_text).tolist()
    y_train = train_df["label"].values
    
    X_val_raw = val_df["text"].apply(clean_text).tolist()
    y_val = val_df["label"].values
    
    X_test_raw = test_df["text"].apply(clean_text).tolist()
    y_test = test_df["label"].values
    
    print(f"Train set: {len(X_train_raw)} (Truthful: {(y_train==0).sum()}, Deceptive: {(y_train==1).sum()})")
    print(f"Val set:   {len(X_val_raw)} (Truthful: {(y_val==0).sum()}, Deceptive: {(y_val==1).sum()})")
    print(f"Test set:  {len(X_test_raw)} (Truthful: {(y_test==0).sum()}, Deceptive: {(y_test==1).sum()})")
    
    # Vectorizers fit strictly on X_train
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        norm='l2'
    )
    X_train_word = word_vec.fit_transform(X_train_raw)
    X_val_word = word_vec.transform(X_val_raw)
    X_test_word = word_vec.transform(X_test_raw)
    
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        min_df=5,
        sublinear_tf=True,
        norm='l2'
    )
    X_train_char = char_vec.fit_transform(X_train_raw)
    X_val_char = char_vec.transform(X_val_raw)
    X_test_char = char_vec.transform(X_test_raw)
    
    X_train_comb = hstack([X_train_word, X_train_char]).tocsr()
    X_val_comb = hstack([X_val_word, X_val_char]).tocsr()
    X_test_comb = hstack([X_test_word, X_test_char]).tocsr()
    
    results = {}
    
    # Model 1: Word-level TF-IDF Linear SVM
    m1 = LinearSVC(C=0.15, max_iter=2000, random_state=42)
    m1.fit(X_train_word, y_train)
    preds1 = m1.predict(X_test_word)
    scores1 = m1.decision_function(X_test_word)
    results["word_tfidf_linear_svm"] = {
        "model_name": "Word-level TF-IDF (1-2 ngrams) + Linear SVM",
        "test_accuracy": round(float(accuracy_score(y_test, preds1)), 4),
        "test_precision": round(float(precision_score(y_test, preds1)), 4),
        "test_recall": round(float(recall_score(y_test, preds1)), 4),
        "test_f1": round(float(f1_score(y_test, preds1)), 4),
        "test_auc": round(float(roc_auc_score(y_test, scores1)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds1).tolist()
    }
    
    # Model 2: Char-level TF-IDF Linear SVM
    m2 = LinearSVC(C=0.15, max_iter=2000, random_state=42)
    m2.fit(X_train_char, y_train)
    preds2 = m2.predict(X_test_char)
    scores2 = m2.decision_function(X_test_char)
    results["char_tfidf_linear_svm"] = {
        "model_name": "Character-level TF-IDF (3-5 ngrams) + Linear SVM",
        "test_accuracy": round(float(accuracy_score(y_test, preds2)), 4),
        "test_precision": round(float(precision_score(y_test, preds2)), 4),
        "test_recall": round(float(recall_score(y_test, preds2)), 4),
        "test_f1": round(float(f1_score(y_test, preds2)), 4),
        "test_auc": round(float(roc_auc_score(y_test, scores2)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds2).tolist()
    }
    
    # Model 3: Combined Word+Char TF-IDF Linear SVM (Our Baseline Champion)
    m3 = LinearSVC(C=0.15, max_iter=2000, random_state=42)
    m3.fit(X_train_comb, y_train)
    preds3 = m3.predict(X_test_comb)
    scores3 = m3.decision_function(X_test_comb)
    results["combined_tfidf_linear_svm"] = {
        "model_name": "Combined Word + Char TF-IDF + Linear SVM (Baseline)",
        "test_accuracy": round(float(accuracy_score(y_test, preds3)), 4),
        "test_precision": round(float(precision_score(y_test, preds3)), 4),
        "test_recall": round(float(recall_score(y_test, preds3)), 4),
        "test_f1": round(float(f1_score(y_test, preds3)), 4),
        "test_auc": round(float(roc_auc_score(y_test, scores3)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds3).tolist()
    }
    
    # Model 4: Calibrated Combined Classifier (5-Fold Platt scaling on training set)
    calibrated_clf = CalibratedClassifierCV(
        estimator=LinearSVC(C=0.15, max_iter=2000, random_state=42),
        method='sigmoid',
        cv=5
    )
    calibrated_clf.fit(X_train_comb, y_train)
    preds4 = calibrated_clf.predict(X_test_comb)
    probs4 = calibrated_clf.predict_proba(X_test_comb)[:, 1]
    results["calibrated_combined_svm"] = {
        "model_name": "Calibrated Combined Word+Char SVM (Platt Scaling via Sigmoid)",
        "test_accuracy": round(float(accuracy_score(y_test, preds4)), 4),
        "test_precision": round(float(precision_score(y_test, preds4)), 4),
        "test_recall": round(float(recall_score(y_test, preds4)), 4),
        "test_f1": round(float(f1_score(y_test, preds4)), 4),
        "test_auc": round(float(roc_auc_score(y_test, probs4)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds4).tolist(),
        "brier_score_loss": round(float(np.mean((probs4 - y_test) ** 2)), 4)
    }
    
    # Validation Domain breakdown
    domain_breakdown = {
        "hotel_review": {
            "status": "Validated (Gold Standard Split)",
            "test_sample_count": len(y_test),
            "test_accuracy": results["calibrated_combined_svm"]["test_accuracy"],
            "test_f1": results["calibrated_combined_svm"]["test_f1"]
        },
        "product_review": {
            "status": "Supported / Extensible pipeline (Awaiting domain ground truth)",
            "test_sample_count": 0,
            "test_accuracy": None,
            "test_f1": None
        },
        "email_message": {
            "status": "Supported / Extensible pipeline (Awaiting domain ground truth)",
            "test_sample_count": 0,
            "test_accuracy": None,
            "test_f1": None
        },
        "social_comment": {
            "status": "Supported / Extensible pipeline (Awaiting domain ground truth)",
            "test_sample_count": 0,
            "test_accuracy": None,
            "test_f1": None
        },
        "personal_statement": {
            "status": "Supported / Extensible pipeline (Awaiting domain ground truth)",
            "test_sample_count": 0,
            "test_accuracy": None,
            "test_f1": None
        }
    }
    
    full_report = {
        "models_comparison": results,
        "selected_model": "calibrated_combined_svm",
        "domain_evaluation": domain_breakdown,
        "cross_domain_findings": (
            "Cross-domain evaluation requires labeled non-hotel ground truth corpora. "
            "Until external datasets are imported through data_pipeline.py, non-hotel predictions "
            "are classified under general lexical and n-gram patterns, and marked as unvalidated in UI/API."
        )
    }
    
    # Save artifacts for production use
    joblib.dump(m3, "general_linear_svm.pkl")
    joblib.dump(calibrated_clf, "general_calibrated_clf.pkl")
    joblib.dump(word_vec, "general_word_vectorizer.pkl")
    joblib.dump(char_vec, "general_char_vectorizer.pkl")
    
    with open("general_model_metrics.json", "w") as f:
        json.dump(full_report, f, indent=2)
        
    print("Saved general model artifacts and general_model_metrics.json!")
    print(json.dumps(full_report, indent=2))

if __name__ == "__main__":
    run_evaluation()
