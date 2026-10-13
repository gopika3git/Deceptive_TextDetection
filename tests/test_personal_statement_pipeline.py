"""
tests/test_personal_statement_pipeline.py
Tests for the Personal Statement Dataset Expansion (Pérez-Rosas & Mihalcea 2015):
1. Raw archive and standardized artifacts existence.
2. OpenDomainDeceptionImporter schema conformity, label mapping, and author ID extraction.
3. Strict zero contributor-leakage and zero text-hash leakage across train, val, and test splits.
4. Zero cross-dataset overlap with hotel reviews.
5. Removal of contradictory/conflicting label duplicate texts.
6. Personal statement model artifact loading and prediction capability.
7. GeneralClassifier domain routing, dynamic caveats, and attribution.
8. FastAPI /predict endpoint integration for personal_statement domain.
9. Multi-domain experiment benchmark metrics artifact validation.
"""

import os
import sys
import json
import pytest
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from data_pipeline import OpenDomainDeceptionImporter, compute_text_hash
from general_classifier import GeneralClassifier

client = TestClient(app)


def test_personal_statement_artifacts_exist():
    """Verify all required personal statement artifacts exist."""
    assert os.path.exists("openDeception.2015.tar.gz")
    assert os.path.exists("personal_statements_raw.csv")
    assert os.path.exists("personal_statements_corpus.csv")
    assert os.path.exists("personal_statements_split_train.csv")
    assert os.path.exists("personal_statements_split_val.csv")
    assert os.path.exists("personal_statements_split_test.csv")
    assert os.path.exists("dataset_audit_personal_statements.json")
    assert os.path.exists("personal_statement_linear_svm.pkl")
    assert os.path.exists("personal_statement_calibrated_clf.pkl")
    assert os.path.exists("personal_statement_word_vectorizer.pkl")
    assert os.path.exists("personal_statement_char_vectorizer.pkl")
    assert os.path.exists("multi_domain_experiments_metrics.json")


def test_importer_schema_and_metadata():
    """Verify OpenDomainDeceptionImporter maps raw records to the standard project schema."""
    importer = OpenDomainDeceptionImporter("personal_statements_raw.csv")
    df = importer.load_and_standardize()

    expected_cols = [
        "text", "label", "domain", "source_dataset",
        "source_label", "author_id", "date", "metadata"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 7168

    # Verify domain and provenance
    assert (df["domain"] == "personal_statement").all()
    assert (df["source_dataset"] == "perez_rosas_mihalcea_2015").all()

    # Verify label mapping: truth -> 0, lie -> 1
    assert set(df["label"].unique()) == {0, 1}
    assert (df[df["source_label"] == "truth"]["label"] == 0).all()
    assert (df[df["source_label"] == "lie"]["label"] == 1).all()

    # Verify 512 authors extracted
    assert df["author_id"].nunique() == 512


def test_contributor_grouped_separation_zero_leakage():
    """Verify strictly 0% contributor and text hash overlap between train, val, and test splits."""
    train_df = pd.read_csv("personal_statements_split_train.csv")
    val_df = pd.read_csv("personal_statements_split_val.csv")
    test_df = pd.read_csv("personal_statements_split_test.csv")

    train_authors = set(train_df["author_id"])
    val_authors = set(val_df["author_id"])
    test_authors = set(test_df["author_id"])

    assert len(train_authors.intersection(val_authors)) == 0, "Contributor leakage: Train and Val overlap"
    assert len(train_authors.intersection(test_authors)) == 0, "Contributor leakage: Train and Test overlap"
    assert len(val_authors.intersection(test_authors)) == 0, "Contributor leakage: Val and Test overlap"

    train_hashes = set(train_df["text"].apply(compute_text_hash))
    val_hashes = set(val_df["text"].apply(compute_text_hash))
    test_hashes = set(test_df["text"].apply(compute_text_hash))

    assert len(train_hashes.intersection(val_hashes)) == 0, "Text hash leakage: Train and Val overlap"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Text hash leakage: Train and Test overlap"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Text hash leakage: Val and Test overlap"


def test_zero_cross_dataset_overlap_with_hotels():
    """Verify zero overlap between personal statements and hotel review datasets."""
    hotel_corpus = pd.read_csv("standardized_multi_domain_corpus.csv")
    ps_corpus = pd.read_csv("personal_statements_corpus.csv")

    hotel_hashes = set(hotel_corpus["text"].apply(compute_text_hash))
    ps_hashes = set(ps_corpus["text"].apply(compute_text_hash))

    overlap = hotel_hashes.intersection(ps_hashes)
    assert len(overlap) == 0, f"Cross-dataset leakage found: {len(overlap)} overlapping samples"


def test_conflicting_label_removal():
    """Verify that contradictory duplicate sentences are removed from the clean corpus."""
    audit_path = "dataset_audit_personal_statements.json"
    with open(audit_path, "r") as f:
        audit = json.load(f)

    assert audit["conflicting_duplicate_labels"] == 56
    assert audit["conflicting_rows_dropped"] > 0

    # Ensure no conflicting text hashes exist in the cleaned corpus
    ps_corpus = pd.read_csv("personal_statements_corpus.csv")
    hashes = ps_corpus["text"].apply(compute_text_hash)
    assert hashes.duplicated().sum() == 0, "Duplicates found in cleaned personal statement corpus"


def test_personal_statement_models_load_and_predict():
    """Verify specialized personal statement models load and predict both classes."""
    svm_model = joblib.load("personal_statement_linear_svm.pkl")
    calibrated_clf = joblib.load("personal_statement_calibrated_clf.pkl")
    word_vec = joblib.load("personal_statement_word_vectorizer.pkl")
    char_vec = joblib.load("personal_statement_char_vectorizer.pkl")

    test_samples = [
        "I was born with four feet and I never sleep.",
        "I graduated from college and work as an analyst."
    ]

    from scipy.sparse import hstack
    X = hstack([word_vec.transform(test_samples), char_vec.transform(test_samples)]).tocsr()

    preds = svm_model.predict(X)
    probs = calibrated_clf.predict_proba(X)

    assert len(preds) == 2
    assert probs.shape == (2, 2)
    assert set(preds).issubset({0, 1})


def test_general_classifier_domain_routing():
    """Verify GeneralClassifier routes to personal statement model and returns accurate caveat."""
    clf = GeneralClassifier()

    sample_text = "I have never lost a game in my entire life."
    res = clf.classify_and_explain(sample_text, domain="personal_statement")

    assert res["domain"] == "personal_statement"
    assert res["is_domain_validated"] is True
    assert "Pérez-Rosas & Mihalcea 2015" in res["domain_caveat"]
    assert "62.55%" in res["domain_caveat"]
    assert res["label"] in ["Truthful", "Deceptive"]
    assert isinstance(res["calibrated_probability"], float)
    assert len(res["word_highlights"]) > 0


def test_api_predict_personal_statement_endpoint():
    """Verify FastAPI /predict handles personal_statement domain correctly."""
    response = client.post("/predict", json={
        "text": "I can bench press 600 pounds without training.",
        "domain": "personal_statement",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["domain"] == "personal_statement"
    assert "Validated" in data["domain_status"]
    assert "62.55%" in data["domain_caveat"]
    assert data["label"] in ["Truthful", "Deceptive"]


def test_multi_domain_experiments_metrics_artifact():
    """Verify multi-domain experiments metrics file contains verified 4-tier benchmark results."""
    with open("multi_domain_experiments_metrics.json", "r") as f:
        metrics = json.load(f)

    exps = metrics["experiments"]
    assert "exp1_in_domain_personal_statements" in exps
    assert "exp2_cross_domain_hotel_to_ps" in exps
    assert "exp3_cross_domain_ps_to_hotel" in exps
    assert "exp4_joint_multi_domain" in exps

    exp1 = exps["exp1_in_domain_personal_statements"]
    assert exp1["accuracy"] == 0.6255
    assert exp1["sample_count"] == 1370

    exp2 = exps["exp2_cross_domain_hotel_to_ps"]
    assert exp2["accuracy"] < 0.50, "Hotel model on PS should exhibit chance performance"

    exp4 = exps["exp4_joint_multi_domain"]
    assert exp4["hotel_evaluation"]["accuracy"] >= 0.84
    assert exp4["personal_statement_evaluation"]["accuracy"] >= 0.62


def test_model_specific_explanations_and_intercept():
    """Verify that feature attributions, vocabulary, and intercepts strictly match the active model."""
    clf = GeneralClassifier()
    sample = "I worked as an analyst for three years and completed the project."

    res_hotel = clf.classify_and_explain(sample, domain="hotel_review")
    res_ps = clf.classify_and_explain(sample, domain="personal_statement")

    # Hotel intercept vs PS intercept must be model-specific
    assert res_hotel["intercept"] == round(clf.intercept, 4)
    assert res_ps["intercept"] == round(clf.ps_intercept, 4)
    assert res_hotel["intercept"] != res_ps["intercept"]

    # Model confidence must not be described as lying
    assert "resemblance" in res_ps["scientific_disclaimer"].lower()
    assert "veracity" in res_ps["scientific_disclaimer"].lower()


def test_combined_analyzer_with_personal_statement_and_sample_handling():
    """Verify combined analyzer routes domain correctly, keeps engines distinct, and handles sample volume honestly."""
    from combined_analyzer import CombinedAnalyzer
    from profile_manager import ProfileManager

    analyzer = CombinedAnalyzer()
    mgr = ProfileManager()
    
    # 1. No profile supplied: Engine A executes, Engine B notes absence
    sample = "I went to school in California and graduated in 2020."
    res_no_prof = analyzer.analyze(sample, domain="personal_statement", profile_id=None)
    assert res_no_prof["engine_a_general_classification"]["domain"] == "personal_statement"
    assert res_no_prof["engine_b_personal_stylometry"] is None
    assert "No personal historical baseline" in res_no_prof["combined_synthesis"]["synthesis_narrative"]
    assert res_no_prof["combined_synthesis"]["concordance_type"] == "Single Engine"

    # 2. Empty profile with 0 samples: handled honestly as incomplete baseline
    empty_prof = mgr.create_profile("Empty Test Profile", description="Test", user_authorization_confirmed=True)
    res_empty = analyzer.analyze(sample, domain="personal_statement", profile_id=empty_prof["id"])
    assert res_empty["engine_b_personal_stylometry"] is not None
    assert "error" in res_empty["engine_b_personal_stylometry"]
    assert "could not be evaluated" in res_empty["combined_synthesis"]["synthesis_narrative"]
    assert "Baseline Incomplete" in res_empty["combined_synthesis"]["concordance_type"]

    # Clean up test profile
    mgr.delete_profile(empty_prof["id"])


def test_api_metrics_includes_multi_domain():
    """Verify /metrics returns both baseline metrics and multi_domain_experiments matrix."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "multi_domain_experiments" in data
    assert "exp1_in_domain_personal_statements" in data["multi_domain_experiments"]["experiments"]


def test_api_combined_endpoint_routes_domain():
    """Verify /api/analyze-combined endpoint passes domain to Engine A."""
    response = client.post("/api/analyze-combined", json={
        "text": "I traveled across Europe after graduating from high school.",
        "domain": "personal_statement"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["engine_a_general_classification"]["domain"] == "personal_statement"
    assert "Pérez-Rosas & Mihalcea 2015" in data["engine_a_general_classification"]["domain_caveat"]

