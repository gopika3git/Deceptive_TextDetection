"""
tests/test_investigative_workflow.py
Verification Test Suite for the Target-Author Investigative Workflow:
1. Standalone Engine A general classification without an author profile.
2. Target-author profile creation with explicit authorization confirmation.
3. Baseline data sufficiency and operational readiness checking (/readiness endpoint).
4. Prevention of using unverified/inappropriate baselines (strict separation per author).
5. Labeling and distinction between demonstration data and real target author profiles.
6. Engine B stylometric comparison against the selected target author's genuine baseline.
7. Dual-engine combined synthesis separation (no arbitrary score merging).
8. Sample mutation, deletion, and permanent data purging.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, profile_mgr

client = TestClient(app)

TEST_PROFILE_ID = "test_target_subject_alpha"


@pytest.fixture(autouse=True)
def cleanup_test_profile():
    """Ensure clean test environment before and after tests."""
    yield
    # Cleanup after test if exists
    if profile_mgr.get_profile(TEST_PROFILE_ID):
        profile_mgr.delete_profile(TEST_PROFILE_ID)


def test_general_classification_without_profile():
    """Verify Engine A runs standalone on submitted text without any author profile."""
    resp = client.post("/predict", json={
        "text": "The hotel receptionist greeted us immediately and handed us keys to a quiet room on the third floor.",
        "domain": "hotel_review"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["label"] in ["Truthful", "Deceptive"]
    assert "decision_score" in data
    assert "calibrated_probability" in data
    assert "word_highlights" in data


def test_profile_creation_requires_authorization():
    """Verify target author profile creation fails without explicit authorization."""
    resp = client.post("/api/profiles", json={
        "name": "Target Subject Alpha",
        "description": "Witness testimony analysis",
        "notes": "Case #402",
        "user_authorization_confirmed": False
    })
    assert resp.status_code == 400
    assert "Authorization confirmation is required" in resp.json()["detail"]


def test_target_author_profile_creation_and_demo_tagging():
    """Verify target author profile is created with is_demo=False while demo_analyst has is_demo=True."""
    # Create custom target author profile
    resp = client.post("/api/profiles", json={
        "name": "Target Subject Alpha",
        "description": "Subject in Case 402",
        "notes": "Verified correspondence",
        "user_authorization_confirmed": True
    })
    assert resp.status_code == 200
    prof = resp.json()["profile"]
    prof_id = prof["id"]
    assert prof["name"] == "Target Subject Alpha"
    assert prof["is_demo"] is False

    # Check profiles list
    list_resp = client.get("/api/profiles")
    assert list_resp.status_code == 200
    profiles = list_resp.json()["profiles"]

    demo_p = next((p for p in profiles if p["id"] == "demo_analyst"), None)
    target_p = next((p for p in profiles if p["id"] == prof_id), None)

    assert demo_p is not None
    assert demo_p["is_demo"] is True

    assert target_p is not None
    assert target_p["is_demo"] is False

    # Clean up created profile
    profile_mgr.delete_profile(prof_id)


def test_baseline_readiness_insufficient_samples():
    """Verify baseline readiness reports incomplete when < 3 samples or < 200 words."""
    # Create profile with only 1 sample (under 200 words)
    prof = profile_mgr.create_profile(
        name="Subject Few Samples",
        description="Testing insufficient baseline",
        user_authorization_confirmed=True,
        profile_id=TEST_PROFILE_ID,
        initial_samples=[{
            "text": "This is a single short sample of normal correspondence that has about fifteen words.",
            "context": "professional_email",
            "is_representative": True
        }]
    )

    resp = client.get(f"/api/profiles/{TEST_PROFILE_ID}/readiness")
    assert resp.status_code == 200
    ready_data = resp.json()

    assert ready_data["status"] == "success"
    assert ready_data["is_operationally_ready"] is False
    assert ready_data["readiness_status"] == "Needs More Writing Samples"
    assert ready_data["min_samples_needed"] >= 2
    assert ready_data["min_words_needed"] > 0
    assert "operational readiness" in ready_data["scientific_limitation_note"].lower()


def test_baseline_readiness_sufficient_samples():
    """Verify baseline readiness reports Operationally Ready when >= 3 samples and >= 200 words."""
    # Create profile with 3 substantial samples totalling > 200 words
    samples = [
        {
            "text": (
                "Regarding our quarterly financial review, we have concluded the comprehensive revenue reconciliation "
                "across all regional branches and international operating subsidiaries. Operational expenses "
                "declined by four point two percent following the infrastructure optimization and cloud migration "
                "initiative completed last spring. The overall departmental performance continues to track favorably "
                "against annual targets and board projections. We expect our audit documentation to be submitted to "
                "the finance committee well before the upcoming quarterly close deadline."
            ),
            "context": "professional_email",
            "is_representative": True
        },
        {
            "text": (
                "The vendor compliance checklist has been thoroughly examined and redlined by outside legal counsel. "
                "All indemnity clauses in section eight have been amended to reflect our enterprise security standards "
                "and regulatory governance policies. Please review the attached contract markup ahead of the committee "
                "meeting tomorrow afternoon so we can resolve any remaining discrepancies before formal contract signing. "
                "I have also notified procurement to hold off on vendor disbursement until legal approval is finalized."
            ),
            "context": "professional_email",
            "is_representative": True
        },
        {
            "text": (
                "Following up on the customer service latency audit, our engineering teams identified several database "
                "bottlenecks during peak transaction windows on weekdays. We are implementing an asynchronous indexing "
                "strategy to reduce queue times and ensure high system responsiveness across all customer-facing services. "
                "We expect preliminary performance benchmarks by Friday afternoon and will circulate the complete report "
                "to stakeholders immediately following our scheduled engineering retro."
            ),
            "context": "professional_email",
            "is_representative": True
        }
    ]

    profile_mgr.create_profile(
        name="Subject Ready Baseline",
        description="Fully established baseline",
        user_authorization_confirmed=True,
        profile_id=TEST_PROFILE_ID,
        initial_samples=samples
    )

    resp = client.get(f"/api/profiles/{TEST_PROFILE_ID}/readiness")
    assert resp.status_code == 200
    ready_data = resp.json()

    assert ready_data["status"] == "success"
    assert ready_data["is_operationally_ready"] is True
    assert ready_data["readiness_status"] == "Operationally Ready for Comparison"
    assert ready_data["min_samples_needed"] == 0
    assert ready_data["min_words_needed"] == 0
    assert ready_data["total_words"] >= 200
    assert ready_data["sample_count"] == 3


def test_target_author_stylometry_comparison():
    """Verify Engine B compares submitted text specifically against the target author's baseline."""
    # Use demo_analyst as target subject
    submitted_text = (
        "Good morning colleagues. I have compiled the quarterly performance metrics and updated "
        "the executive presentation deck. Please send any feedback prior to the Friday board review."
    )

    resp = client.post("/api/compare-personal", json={
        "text": submitted_text,
        "profile_id": "demo_analyst"
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["profile_id"] == "demo_analyst"
    assert "comparison" in data
    comp = data["comparison"]
    assert "alignment_index" in comp
    assert isinstance(comp["alignment_index"], (int, float))
    assert "top_shifted_features" in comp
    assert "shifted_features_count" in comp
    assert "stable_features" in comp
    assert "alternative_explanations" in comp


def test_combined_analysis_without_author_baseline():
    """Verify combined analysis gracefully handles no author baseline (runs Engine A only)."""
    submitted_text = "The room was spotless, the bed was very comfortable, and checkout was effortless."
    resp = client.post("/api/analyze-combined", json={
        "text": submitted_text,
        "domain": "hotel_review",
        "profile_id": None
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["engine_a_general_classification"] is not None
    assert data["engine_b_personal_stylometry"] is None
    assert "Single Engine" in data["combined_synthesis"]["concordance_type"]
    assert "No personal historical baseline was provided" in data["combined_synthesis"]["synthesis_narrative"]
    assert "Independent Metric Rule" in data["combined_synthesis"]["methodology_reminder"]


def test_combined_analysis_with_target_author_baseline():
    """Verify combined analysis separates general classification from author stylometric comparison."""
    submitted_text = (
        "I was deeply impressed with the prompt customer service and the clarity of the financial breakdown. "
        "The manager personally checked on our satisfaction before the final contract signing."
    )
    resp = client.post("/api/analyze-combined", json={
        "text": submitted_text,
        "domain": "hotel_review",
        "profile_id": "demo_analyst"
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["engine_a_general_classification"] is not None
    assert data["engine_b_personal_stylometry"] is not None
    assert "alignment_index" in data["engine_b_personal_stylometry"]
    assert "concordance_type" in data["combined_synthesis"]
    assert "Independent Metric Rule" in data["combined_synthesis"]["methodology_reminder"]
    # Check that scores are kept independent and not blended into an invented percentage
    assert "combined_deception_percentage" not in data["combined_synthesis"]


def test_target_author_sample_deletion_purges_text():
    """Verify privacy guarantee: deleting a sample permanently updates the target author baseline."""
    prof = profile_mgr.create_profile(
        name="Privacy Purge Subject",
        user_authorization_confirmed=True,
        profile_id=TEST_PROFILE_ID,
        initial_samples=[{
            "text": "Sample text number one for privacy verification testing in the target subject pipeline.",
            "context": "professional_email"
        }]
    )
    sample_id = prof["samples"][0]["id"]

    # Delete sample
    del_resp = client.delete(f"/api/profiles/{TEST_PROFILE_ID}/samples/{sample_id}")
    assert del_resp.status_code == 200
    updated_prof = del_resp.json()["profile"]
    assert len(updated_prof["samples"]) == 0
    assert updated_prof["baseline"]["sample_count"] == 0
    assert updated_prof["baseline"]["is_operationally_ready"] is False


def test_regression_combined_analysis_moderate_stylistic_divergence_no_contradiction():
    """Regression test: verify Moderate Stylistic Divergence is never summarized as matching baseline."""
    # Text known to produce Moderate Stylistic Divergence against demo_analyst
    test_text = "The hotel was lovely and clean, but the breakfast options were somewhat limited for vegetarian guests."
    resp = client.post("/api/analyze-combined", json={
        "text": test_text,
        "domain": "hotel_review",
        "profile_id": "demo_analyst"
    })
    assert resp.status_code == 200
    data = resp.json()
    b = data["engine_b_personal_stylometry"]
    synth = data["combined_synthesis"]

    assert b["overall_assessment"] == "Moderate Stylistic Divergence"
    assert 55.0 <= b["alignment_index"] < 80.0
    # Crucial assertion: Narrative must NOT claim it matches the baseline
    assert "matches" not in synth["synthesis_narrative"].lower() or "not match" in synth["synthesis_narrative"].lower()
    assert "moderate stylistic divergence" in synth["synthesis_narrative"].lower()
    assert "Moderate Stylistic Divergence" in synth["concordance_type"]


def test_regression_engine_a_routing_and_malformed_edge_cases():
    """Regression test: ensure Engine A handles extreme, repeated, and malformed inputs gracefully."""
    edge_cases = [
        "a",
        "!??.. ,,, !!!",
        "12345 67890 99999",
        "hotel " * 200,
        "The luxury resort provided exceptional room service and friendly staff. " * 300,
        "Great stay! 🏨✨ Loved the pool 😍! Déjà vu à l'hôtel été 🌟",
        "<script>alert('xss')</script><div>Beautiful hotel room with <b>balcony</b>.</div>",
        "Line 1\r\nLine 2\t\tLine 3\x00Null character ignored",
    ]
    for edge_input in edge_cases:
        resp = client.post("/predict", json={"text": edge_input, "domain": "hotel_review"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["label"] in ["Truthful", "Deceptive"]
        assert "decision_score" in data


def test_regression_baseline_readiness_exact_word_threshold():
    """Regression test: ensure baseline requires BOTH sample_count >= 3 AND total_words >= 200."""
    reg_pid = "reg_thresh_test_profile"
    profile_mgr.delete_profile(reg_pid)

    # 3 samples but each only 10 words -> total ~30 words (< 200)
    samples_few_words = [
        {"text": "Short update on the project kickoff meeting for the team.", "context": "email"},
        {"text": "Confirming the schedule for next week client presentation review.", "context": "email"},
        {"text": "Thank you for the quick reply on the contract draft.", "context": "email"}
    ]
    prof = profile_mgr.create_profile(
        name="Threshold Test Subject",
        user_authorization_confirmed=True,
        profile_id=reg_pid,
        initial_samples=samples_few_words
    )
    assert prof["baseline"]["sample_count"] == 3
    assert prof["baseline"]["total_word_count"] < 200
    assert prof["baseline"]["is_operationally_ready"] is False
    assert "Needs More Writing Samples" in prof["baseline"]["readiness_status"]

    profile_mgr.delete_profile(reg_pid)


def test_regression_invalid_and_missing_profile_handling():
    """Regression test: verify non-existent and path-traversal profile requests are safely rejected."""
    # 1. Non-existent profile in /api/profiles/{id}
    resp_404 = client.get("/api/profiles/non_existent_profile_xyz_123")
    assert resp_404.status_code == 404

    # 2. Path-traversal attempt in profile ID
    resp_trav = client.get("/api/profiles/../../etc/passwd")
    assert resp_trav.status_code == 404

    # 3. Non-existent profile in /api/compare-personal
    resp_comp = client.post("/api/compare-personal", json={
        "text": "This is a legitimate verification text for comparative stylometry.",
        "profile_id": "non_existent_profile_xyz_123"
    })
    assert resp_comp.status_code == 404


def test_regression_demo_profile_deletion_forbidden():
    """Regression test: verify built-in demo analyst profile is strictly protected against deletion."""
    resp = client.delete("/api/profiles/demo_analyst")
    assert resp.status_code == 403
    assert "demo_analyst" in resp.json()["detail"]
    assert profile_mgr.get_profile("demo_analyst") is not None


def test_regression_empty_sanitized_profile_id_rejected():
    """Regression test: verify sanitized-empty profile IDs do not match .json or crash."""
    assert profile_mgr.get_profile("../..") is None
    assert profile_mgr.delete_profile("../..") is False
    assert profile_mgr.get_profile("") is None

