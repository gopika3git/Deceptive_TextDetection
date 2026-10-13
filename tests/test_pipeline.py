"""
Comprehensive Verification and Regression Test Suite
Covers:
1. Model artifact loading & version compatibility
2. Zero data leakage verification
3. Both class prediction capability
4. Empty & edge-case input handling
5. Explanation-prediction mathematical alignment
6. Feedback logging and validation
7. API endpoints contract verification
8. Frontend asset accessibility
"""

import os
import sys
import json
import pytest
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app import app
from explainer import DeceptionExplainer, clean_text

client = TestClient(app)

def test_artifacts_exist_and_load():
    """Verify all required model artifacts and metadata exist and load."""
    assert os.path.exists("final_svm_model.pkl")
    assert os.path.exists("final_word_vectorizer.pkl")
    assert os.path.exists("final_char_vectorizer.pkl")
    assert os.path.exists("model_metadata.json")
    
    model = joblib.load("final_svm_model.pkl")
    word_vec = joblib.load("final_word_vectorizer.pkl")
    char_vec = joblib.load("final_char_vectorizer.pkl")
    
    assert hasattr(model, "predict")
    assert hasattr(model, "decision_function")
    assert hasattr(word_vec, "transform")
    assert hasattr(char_vec, "transform")

def test_data_leakage_audit():
    """Verify strictly 0% leakage between train_leakfree and test sets."""
    assert os.path.exists("train_leakfree.csv")
    assert os.path.exists("test_original_untouched.csv")
    assert os.path.exists("test_deduplicated.csv")
    
    train_df = pd.read_csv("train_leakfree.csv")
    test_untouched = pd.read_csv("test_original_untouched.csv")
    test_dedup = pd.read_csv("test_deduplicated.csv")
    
    train_texts = set(train_df["text_clean"].str.lower().str.strip())
    test_untouched_texts = set(test_untouched["text_clean"].str.lower().str.strip())
    test_dedup_texts = set(test_dedup["text_clean"].str.lower().str.strip())
    
    overlap_untouched = train_texts.intersection(test_untouched_texts)
    overlap_dedup = train_texts.intersection(test_dedup_texts)
    
    assert len(overlap_untouched) == 0, f"Leakage detected with untouched test: {len(overlap_untouched)}"
    assert len(overlap_dedup) == 0, f"Leakage detected with deduplicated test: {len(overlap_dedup)}"

def test_prediction_both_classes():
    """Verify that the model successfully predicts both Truthful and Deceptive."""
    explainer = DeceptionExplainer()
    
    # Highly exaggerated promotional text
    sample_deceptive = "I had the most extraordinary experience in the entire world! The staff was the warmest and most wonderful angels, magnificent luxurious rooms."
    # Specific granular experiential review
    sample_truthful = "We stayed on the fourth floor for two nights. The elevator took some time on Saturday morning. The shower had decent water pressure and the front desk gave us directions to Michigan Avenue."
    
    res_dec = explainer.predict_and_explain(sample_deceptive)
    res_tru = explainer.predict_and_explain(sample_truthful)
    
    # Verify predictions
    assert res_dec["label"] in ["Truthful", "Deceptive"]
    assert res_tru["label"] in ["Truthful", "Deceptive"]
    
    # Test set predictions distribution check
    test_df = pd.read_csv("test_deduplicated.csv")
    X_w = explainer.word_vectorizer.transform(test_df["text_clean"])
    X_c = explainer.char_vectorizer.transform(test_df["text_clean"])
    from scipy.sparse import hstack
    X = hstack([X_w, X_c]).tocsr()
    preds = explainer.model.predict(X)
    
    unique_preds = set(preds)
    assert 0 in unique_preds, "Model never predicts Truthful (0)"
    assert 1 in unique_preds, "Model never predicts Deceptive (1)"

def test_empty_and_edge_case_inputs():
    """Verify graceful handling of empty strings, whitespace, and special characters."""
    explainer = DeceptionExplainer()
    
    res_empty = explainer.predict_and_explain("")
    assert res_empty["label"] == "Unknown"
    assert "error" in res_empty
    
    res_spaces = explainer.predict_and_explain("     \n\t   ")
    assert res_spaces["label"] == "Unknown"
    
    res_punct = explainer.predict_and_explain("!@#$%^&*()_+")
    assert res_punct["label"] in ["Truthful", "Deceptive", "Unknown"]

def test_explanation_matches_prediction():
    """Verify mathematical consistency between decision score, label, and feature contributions."""
    explainer = DeceptionExplainer()
    review = "Clean room, comfortable bed, but noisy street outside. The front desk was helpful."
    res = explainer.predict_and_explain(review)
    
    score = res["decision_score"]
    label = res["label"]
    
    if score > 0:
        assert label == "Deceptive", f"Score {score} > 0 must result in Deceptive, got {label}"
    else:
        assert label == "Truthful", f"Score {score} <= 0 must result in Truthful, got {label}"
        
    for feat in res["top_deceptive_features"]:
        assert feat["contribution"] > 0, "Deceptive cue must have positive contribution"
        
    for feat in res["top_truthful_features"]:
        assert feat["contribution"] < 0, "Truthful cue must have negative contribution"

def test_api_health_endpoint():
    """Verify /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["engine_a_loaded"] is True
    assert "timestamp" in data
    assert "engine_b_profiles_count" in data
    assert data["engine_b_profiles_count"] >= 0

def test_api_predict_endpoint():
    """Verify /predict endpoint success and error handling."""
    # Valid call
    response = client.post("/predict", json={
        "text": "Great stay, quiet room, friendly check-in.",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "label" in data
    assert "decision_score" in data
    assert "word_highlights" in data
    assert len(data["word_highlights"]) > 0
    
    # Empty call (expect 400)
    err_resp = client.post("/predict", json={"text": "   "})
    assert err_resp.status_code == 400

def test_api_explain_endpoint():
    """Verify /explain endpoint."""
    response = client.post("/explain", json={
        "text": "Wonderful hotel with great location and spacious rooms.",
        "top_k": 6
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "top_deceptive_features" in data
    assert "top_truthful_features" in data

def test_api_metrics_endpoint():
    """Verify dynamic /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert ("metrics" in data or "metadata" in data)
    if "metadata" in data:
        meta = data["metadata"]
        assert "test_results_untouched" in meta
        assert meta["test_results_untouched"]["accuracy"] > 0.85
    elif "metrics" in data:
        metrics = data["metrics"]
        assert "models_comparison" in metrics
        assert metrics["models_comparison"]["calibrated_combined_svm"]["test_accuracy"] > 0.85

def test_api_feedback_submission():
    """Verify feedback validation and storage."""
    # Valid feedback
    resp = client.post("/feedback", json={
        "review_text": "A test review text",
        "predicted_label": "Truthful",
        "user_label": "Truthful",
        "comments": "Accurate prediction"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    
    # Invalid user label
    bad_resp = client.post("/feedback", json={
        "review_text": "A test review text",
        "predicted_label": "Truthful",
        "user_label": "InvalidLabel123",
        "comments": "Test"
    })
    assert bad_resp.status_code == 400

def test_frontend_page_and_assets():
    """Verify UI index page and static assets return 200."""
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "VeritasAI" in res_index.text
    
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    
    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
