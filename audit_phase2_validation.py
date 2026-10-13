"""
audit_phase2_validation.py
Comprehensive validation and reliability audit for Phase 2:
1. Engine A Routing & Stress/Malformed Input Testing
2. Engine B Target-Author Stylometry & Baseline Sufficiency
3. Combined Analysis Narrative Consistency & Metric Separation
4. Profile Persistence & Sample Mutability
5. Model Reliability Evaluation on Labeled Test Splits
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss

from general_classifier import GeneralClassifier, clean_text
from stylometry_engine import StylometricFeatureExtractor, BaselineAggregator, StylometricComparator
from profile_manager import ProfileManager
from combined_analyzer import CombinedAnalyzer

def audit_engine_a():
    print("=" * 60)
    print("AUDIT 1: ENGINE A ROUTING & INPUT ROBUSTNESS")
    print("=" * 60)
    clf = GeneralClassifier()
    
    # 1. Routing between hotel_review and personal_statement
    hotel_text = "The room was spotless and the staff at the front desk was very polite and helpful throughout our stay."
    ps_text = "I have never traveled to Europe because my family could not afford overseas vacations."
    
    res_hotel = clf.classify_and_explain(hotel_text, domain="hotel_review")
    res_ps = clf.classify_and_explain(ps_text, domain="personal_statement")
    
    print(f"Hotel Review: label={res_hotel['label']}, prob={res_hotel['calibrated_probability']}, intercept={res_hotel['intercept']}")
    print(f"Personal Statement: label={res_ps['label']}, prob={res_ps['calibrated_probability']}, intercept={res_ps['intercept']}")
    
    # Verify different models were used
    assert res_hotel["intercept"] != res_ps["intercept"], "Hotel and Personal Statement should use domain-specific models"
    assert res_hotel["domain"] == "hotel_review"
    assert res_ps["domain"] == "personal_statement"
    assert res_hotel["is_domain_validated"] is True
    assert res_ps["is_domain_validated"] is True
    
    # Test unvalidated domain transfer
    res_unval = clf.classify_and_explain("Sample test", domain="email_message")
    assert res_unval["is_domain_validated"] is False
    assert "Notice: The selected domain" in res_unval["domain_caveat"]
    print("Domain routing verified: Hotel Review, Personal Statement, and Unvalidated Transfer.")

    # 2. Stress & Edge Case Inputs
    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   \n\t  "),
        ("Single char", "a"),
        ("Two words", "hello world"),
        ("Punctuation only", "!??.. ,,, !!!"),
        ("Numbers only", "12345 67890 99999"),
        ("Repeated single word", "hotel " * 200),
        ("Extreme long text (5000 words)", "The luxury resort provided exceptional room service and friendly staff. " * 500),
        ("Emojis & Unicode", "Great stay! 🏨✨ Loved the pool 😍! Déjà vu à l'hôtel été 🌟"),
        ("HTML tags & Scripts", "<script>alert('xss')</script><div>Beautiful hotel room with <b>balcony</b>.</div>"),
        ("Special control chars & newlines", "Line 1\r\nLine 2\t\tLine 3\x00Null character ignored"),
    ]
    
    for name, input_str in edge_cases:
        try:
            out_h = clf.classify_and_explain(input_str, domain="hotel_review")
            out_ps = clf.classify_and_explain(input_str, domain="personal_statement")
            print(f"  [PASS] Edge case '{name}': Hotel label={out_h.get('label')}, PS label={out_ps.get('label')}")
        except Exception as e:
            print(f"  [FAIL] Edge case '{name}' raised exception: {e}")
            raise e

    print("Engine A Stress Testing PASSED.")

def audit_engine_b():
    print("\n" + "=" * 60)
    print("AUDIT 2: ENGINE B TARGET-AUTHOR STYLOMETRY & READINESS")
    print("=" * 60)
    
    # 1. Zero samples baseline
    base_0 = BaselineAggregator.compute_baseline([])
    assert base_0["is_operationally_ready"] is False
    assert base_0["sample_count"] == 0
    assert base_0["readiness_status"] == "No Samples Added"
    
    comp_0 = StylometricComparator.compare("Test text to compare", base_0)
    assert "error" in comp_0
    print("Zero-sample baseline correctly rejected by comparator.")
    
    # 2. Insufficient baseline (1 sample, 40 words)
    sample_short = [{"id": "s1", "text": "This is a short sample sent by the author to colleagues regarding the project kickoff meeting scheduled for Monday morning.", "is_representative": True}]
    base_1 = BaselineAggregator.compute_baseline(sample_short)
    assert base_1["is_operationally_ready"] is False
    assert "Needs More Writing Samples" in base_1["readiness_status"]
    assert base_1["min_samples_needed"] == 2
    assert base_1["min_words_needed"] > 0
    print(f"1-sample baseline readiness: ready={base_1['is_operationally_ready']}, status='{base_1['readiness_status']}', min_samples_needed={base_1['min_samples_needed']}")
    
    # 3. Sufficient baseline (5 samples, 200+ words)
    sample_texts = [
        "Good morning team, I have reviewed the comprehensive quarterly financial metrics and consolidated all department data into the executive briefing presentation. Overall corporate performance exceeded initial expectations across multiple business units, particularly within enterprise software accounts and recurring support agreements.",
        "Regarding the client steering committee meeting next Tuesday, I prepared the slide deck highlighting key product performance indicators, customer retention statistics, and projected quarterly renewals across enterprise tier accounts. Please review the attached redlines and send feedback before tomorrow afternoon.",
        "Attached is the finalized technical specification document for the cloud platform architecture upgrade planned for the upcoming fiscal sprint. Our senior engineering team conducted thorough benchmarking across system redundancy, data integrity, and strict latency constraints.",
        "Thank you for attending the engineering retrospective sync yesterday. I have captured all core action items and assigned ownership to team leads to ensure seamless delivery and cross-functional coordination over the coming month. Let me know if any adjustments are needed.",
        "Following up on our discussions with the product design department, we have completed user experience testing for the customer management portal. The survey responses indicate high overall satisfaction with the refreshed navigation layout and accelerated response times. We will share the full breakdown during Friday's engineering briefing."
    ]
    samples_suff = [{"id": f"s{i}", "text": t, "is_representative": True} for i, t in enumerate(sample_texts)]
    base_suff = BaselineAggregator.compute_baseline(samples_suff)
    assert base_suff["total_word_count"] >= 200, f"Expected >= 200 words, got {base_suff['total_word_count']}"
    assert base_suff["is_operationally_ready"] is True
    assert base_suff["readiness_status"] == "Operationally Ready for Comparison"
    assert base_suff["min_samples_needed"] == 0
    assert base_suff["min_words_needed"] == 0
    print(f"Sufficient baseline: ready={base_suff['is_operationally_ready']}, status='{base_suff['readiness_status']}', reliability='{base_suff['reliability_rating']}'")
    
    # 4. Compare text with sufficient baseline
    test_similar = "Good morning everyone, I have completed the review of our department deliverables and sent the summary to management."
    comp_sim = StylometricComparator.compare(test_similar, base_suff)
    print(f"Similar text comparison: Alignment={comp_sim['alignment_index']}%, Assessment='{comp_sim['overall_assessment']}'")
    
    test_divergent = "YO BRO WASSUP?? CALL ME RN RN!! 100% CRAZY DUDE!! OMG WTF!!!"
    comp_div = StylometricComparator.compare(test_divergent, base_suff)
    print(f"Divergent text comparison: Alignment={comp_div['alignment_index']}%, Assessment='{comp_div['overall_assessment']}'")
    assert comp_div["alignment_index"] < comp_sim["alignment_index"]
    assert len(comp_div["top_shifted_features"]) > 0

    print("Engine B Stylometry Audit PASSED.")

def audit_combined_analysis():
    print("\n" + "=" * 60)
    print("AUDIT 3: COMBINED ANALYSIS & NARRATIVE CONTRADICTIONS")
    print("=" * 60)
    pm = ProfileManager()
    analyzer = CombinedAnalyzer(profile_manager=pm)
    
    # Check what happens with demo_analyst
    demo = pm.get_profile("demo_analyst")
    print(f"Demo profile samples: {len(demo['samples'])}, ready={demo['baseline']['is_operationally_ready']}")
    
    # Test case: Text that produces Moderate Stylistic Divergence
    # Let's inspect different texts against demo_analyst
    test_candidates = [
        "I stayed here last month and was truly disappointed. The room was not cleaned and the noise was intolerable.",
        "The hotel was lovely and clean, but the breakfast options were somewhat limited for vegetarian guests.",
        "Hi Alex, could you send me the latest metrics document before lunch today?",
        "NOTICE TO ALL STAFF: Mandatory emergency evacuation protocol will commence immediately at 14:00 hours."
    ]
    
    for t in test_candidates:
        res = analyzer.analyze(t, domain="hotel_review", profile_id="demo_analyst")
        b = res["engine_b_personal_stylometry"]
        synth = res["combined_synthesis"]
        print("-" * 50)
        print(f"Text: '{t[:60]}...'")
        print(f"Engine A: {res['engine_a_general_classification']['label']} ({res['engine_a_general_classification']['calibrated_probability']})")
        print(f"Engine B: Alignment={b['alignment_index']}%, Assessment='{b['overall_assessment']}'")
        print(f"Synthesis Concordance: '{synth['concordance_type']}'")
        print(f"Synthesis Narrative: {synth['synthesis_narrative']}")
        
        # Check for contradiction:
        if b["overall_assessment"] == "Moderate Stylistic Divergence" and "matches" in synth["synthesis_narrative"]:
            print(">>> CONTRADICTION DETECTED: Assessment is 'Moderate Stylistic Divergence' but narrative says text 'matches' baseline!")

def audit_profile_persistence():
    print("\n" + "=" * 60)
    print("AUDIT 4: PROFILE PERSISTENCE & DATA ISOLATION")
    print("=" * 60)
    pm = ProfileManager()
    test_pid = "test_audit_temp_prof"
    
    # Ensure cleanup first if exists
    pm.delete_profile(test_pid)
    
    # 1. Create test profile
    prof = pm.create_profile(
        name="Audit Test Subject",
        description="Temporary profile for audit",
        notes="Audit notes",
        user_authorization_confirmed=True,
        profile_id=test_pid
    )
    assert prof["id"] == test_pid
    assert prof["is_demo"] is False
    assert os.path.exists(pm._get_profile_path(test_pid))
    print("Profile creation succeeded and written to disk.")
    
    # 2. Add samples
    s1 = pm.add_sample(test_pid, "This is sample one for the target subject.", context="email")
    s2 = pm.add_sample(test_pid, "This is sample two for the target subject with more detailed writing.", context="email")
    assert len(s2["samples"]) == 2
    print(f"Added 2 samples, baseline sample_count={s2['baseline']['sample_count']}")
    
    # 3. Reload from disk
    pm2 = ProfileManager()
    reloaded = pm2.get_profile(test_pid)
    assert reloaded is not None
    assert len(reloaded["samples"]) == 2
    assert reloaded["name"] == "Audit Test Subject"
    print("Reload from disk verified persistence.")
    
    # 4. Delete sample
    samp_id_to_del = reloaded["samples"][0]["id"]
    updated = pm2.delete_sample(test_pid, samp_id_to_del)
    assert len(updated["samples"]) == 1
    assert updated["baseline"]["sample_count"] == 1
    print("Sample deletion verified: text purged, baseline updated.")
    
    # 5. Delete profile and verify file removed
    deleted = pm2.delete_profile(test_pid)
    assert deleted is True
    assert not os.path.exists(pm._get_profile_path(test_pid))
    print("Profile deletion verified: file permanently removed.")
    
    # 6. Verify demo_analyst was untouched
    demo = pm2.get_profile("demo_analyst")
    assert demo is not None
    assert demo["is_demo"] is True
    print("Demo subject isolation verified: demo_analyst intact.")

def audit_reliability_metrics():
    print("\n" + "=" * 60)
    print("AUDIT 5: RELIABILITY EVALUATION ON LABELED TEST SPLITS")
    print("=" * 60)
    clf = GeneralClassifier()
    
    # 1. Hotel Reviews Test Split
    hotel_test_path = "split_test.csv"
    if os.path.exists(hotel_test_path):
        df_hotel = pd.read_csv(hotel_test_path)
        print(f"Evaluating Hotel Reviews Test Set ({len(df_hotel)} samples)...")
        y_true = df_hotel["label"].values
        preds = []
        probs = []
        for text in df_hotel["text"]:
            res = clf.classify_and_explain(text, domain="hotel_review")
            preds.append(res["label_code"])
            probs.append(res["calibrated_probability"])
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        
        print("\n--- Hotel Reviews Test Set Results ---")
        print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision:        {prec:.4f}")
        print(f"  Recall:           {rec:.4f}")
        print(f"  F1 Score:         {f1:.4f}")
        print(f"  ROC-AUC:          {auc:.4f}")
        print(f"  Brier Score Loss: {brier:.4f}")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # 2. Personal Statements Test Split
    ps_test_path = "personal_statements_split_test.csv"
    if os.path.exists(ps_test_path):
        df_ps = pd.read_csv(ps_test_path)
        print(f"\nEvaluating Personal Statements Test Set ({len(df_ps)} samples)...")
        y_true_ps = df_ps["label"].values
        preds_ps = []
        probs_ps = []
        for text in df_ps["text"]:
            res = clf.classify_and_explain(text, domain="personal_statement")
            preds_ps.append(res["label_code"])
            probs_ps.append(res["calibrated_probability"])
            
        acc_ps = accuracy_score(y_true_ps, preds_ps)
        prec_ps = precision_score(y_true_ps, preds_ps)
        rec_ps = recall_score(y_true_ps, preds_ps)
        f1_ps = f1_score(y_true_ps, preds_ps)
        cm_ps = confusion_matrix(y_true_ps, preds_ps)
        auc_ps = roc_auc_score(y_true_ps, probs_ps)
        brier_ps = brier_score_loss(y_true_ps, probs_ps)
        
        print("\n--- Personal Statements Test Set Results ---")
        print(f"  Accuracy:         {acc_ps:.4f} ({acc_ps*100:.2f}%)")
        print(f"  Precision:        {prec_ps:.4f}")
        print(f"  Recall:           {rec_ps:.4f}")
        print(f"  F1 Score:         {f1_ps:.4f}")
        print(f"  ROC-AUC:          {auc_ps:.4f}")
        print(f"  Brier Score Loss: {brier_ps:.4f}")
        print(f"  Confusion Matrix: TN={cm_ps[0,0]}, FP={cm_ps[0,1]}, FN={cm_ps[1,0]}, TP={cm_ps[1,1]}")

if __name__ == "__main__":
    audit_engine_a()
    audit_engine_b()
    audit_combined_analysis()
    audit_profile_persistence()
    audit_reliability_metrics()
