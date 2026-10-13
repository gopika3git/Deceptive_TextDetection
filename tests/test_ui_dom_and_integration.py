"""
tests/test_ui_dom_and_integration.py
End-to-End Integration and UI Verification Tests:
1. Verifies HTML page structure, DOM elements, and labels.
2. Verifies domain routing across Hotel Reviews and Personal Statements.
3. Verifies model-specific explanations and intercepts.
4. Verifies combined analysis separation, honest sample handling, and synthesis.
5. Verifies multi-domain metrics and empirical benchmarks in UI tables and API.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from general_classifier import GeneralClassifier

client = TestClient(app)


def test_ui_html_structure_and_labels():
    """Verify HTML UI contains accurate non-deceptive labels, domain options, and metric tables."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # 1. Non-misleading probability label
    assert "Model Deception Probability:" in html
    assert "Platt Calibrated Sigmoid — Benchmark Resemblance" in html
    assert "Calibrated Deception Prob:" not in html

    # 2. Domain selectors include both validated domains
    assert '<option value="hotel_review">Hotel Reviews (Gold Standard Validated)</option>' in html
    assert '<option value="personal_statement">Personal Statements & Narratives</option>' in html

    # 3. Combined domain selector contains personal statement
    assert '<option value="personal_statement">Personal Statements & Narratives (Validated)</option>' in html

    # 4. Multi-Domain Validation Matrix reports separate benchmarks
    assert "Hotel Reviews (Dedicated Baseline)" in html
    assert "88.75% Test Acc (0.9555 AUC)" in html
    assert "Personal Statements (Dedicated Model)" in html
    assert "62.55% Test Acc (0.6626 AUC)" in html
    assert "Joint Multi-Domain Model (Hotel &amp; Personal Statements)" in html
    assert "Hotel: 85.00% (0.9307 AUC) | PS: 62.85% (0.6562 AUC)" in html
    assert "Cross-Domain Transfer: Hotel &rarr; Personal Statements" in html
    assert "48.32% Test Acc (0.4957 AUC)" in html
    assert "Cross-Domain Transfer: Personal Statements &rarr; Hotel" in html
    assert "49.69% Test Acc (0.5213 AUC)" in html

    # 5. Leakage Prevention Report table shows 0 leakage for both
    assert "0 author overlap and 0 text hash overlap across splits" in html
    assert "0 text hash overlap between Hotel and Personal Statement corpora" in html

    # 6. Target-Author Investigative Workflow & Onboarding Wizard elements
    assert "modal-onboarding-wizard" in html
    assert "btn-header-onboard" in html
    assert "Investigator Protocol:" in html
    assert "Target-Author Stylometric Comparison" in html
    assert "personal-active-profile-indicator" in html
    assert "combined-active-profile-indicator" in html


def test_end_to_end_hotel_review_routing_and_explanation():
    """Test full pipeline for hotel review prediction and explanation."""
    sample = "Our stay at the hotel was quiet, the room was clean, and the staff helped us with parking."
    resp = client.post("/predict", json={
        "text": sample,
        "domain": "hotel_review",
        "top_k": 5
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["domain"] == "hotel_review"
    assert "88.75%" in data["domain_caveat"]
    assert "Ott et al." in data["domain_caveat"]
    assert data["label"] in ["Truthful", "Deceptive"]
    assert isinstance(data["decision_score"], float)
    assert isinstance(data["calibrated_probability"], float)
    assert len(data["top_deceptive_features"]) <= 5
    assert len(data["top_truthful_features"]) <= 5


def test_end_to_end_personal_statement_routing_and_explanation():
    """Test full pipeline for personal statement prediction and explanation."""
    sample = "I grew up in an apartment in Chicago with my two sisters and attended state college."
    resp = client.post("/predict", json={
        "text": sample,
        "domain": "personal_statement",
        "top_k": 5
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["domain"] == "personal_statement"
    assert "62.55%" in data["domain_caveat"]
    assert "Pérez-Rosas & Mihalcea 2015" in data["domain_caveat"]
    assert "subtle deception cues" in data["domain_caveat"]
    assert data["label"] in ["Truthful", "Deceptive"]
    assert isinstance(data["decision_score"], float)
    assert isinstance(data["calibrated_probability"], float)


def test_end_to_end_combined_analysis_dual_domains():
    """Test combined analysis for both domains with separation of Engine A and Engine B."""
    # Hotel review in combined analysis
    resp_hotel = client.post("/api/analyze-combined", json={
        "text": "Great location near the river, clean bedding, and fast check-in.",
        "domain": "hotel_review"
    })
    assert resp_hotel.status_code == 200
    data_h = resp_hotel.json()
    assert data_h["engine_a_general_classification"]["domain"] == "hotel_review"
    assert "88.75%" in data_h["engine_a_general_classification"]["domain_caveat"]
    assert data_h["engine_b_personal_stylometry"] is None
    assert "Single Engine" in data_h["combined_synthesis"]["concordance_type"]

    # Personal statement in combined analysis
    resp_ps = client.post("/api/analyze-combined", json={
        "text": "I worked as an analyst for five years and completed my master's degree in engineering.",
        "domain": "personal_statement"
    })
    assert resp_ps.status_code == 200
    data_ps = resp_ps.json()
    assert data_ps["engine_a_general_classification"]["domain"] == "personal_statement"
    assert "62.55%" in data_ps["engine_a_general_classification"]["domain_caveat"]
    assert data_ps["engine_b_personal_stylometry"] is None
    assert "Single Engine" in data_ps["combined_synthesis"]["concordance_type"]
