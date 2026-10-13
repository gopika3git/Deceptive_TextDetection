"""
general_classifier.py
Engine A — General Text Deception Classifier.

Provides:
- Combined Word (1-2) + Character (3-5) TF-IDF feature extraction
- Linear SVM decision function score (signed geometric distance from hyperplane)
- Platt-calibrated probability of deception
- Domain validation status reporting (Validated vs Unvalidated General Transfer)
- Feature-level and token-level explanations (w_j * x_j contributions)
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
import numpy as np
import joblib
from scipy.sparse import hstack

def clean_text(text: str) -> str:
    """Preprocess text identically to training pipeline."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class GeneralClassifier:
    """
    Engine A Classifier with calibrated probability estimation,
    domain metadata awareness, and feature-level explanations.
    """
    def __init__(
        self,
        base_svm_path: str = "general_linear_svm.pkl",
        calibrated_clf_path: str = "general_calibrated_clf.pkl",
        word_vec_path: str = "general_word_vectorizer.pkl",
        char_vec_path: str = "general_char_vectorizer.pkl",
        catalog_path: str = "multi_domain_catalog.json",
        metrics_path: str = "general_model_metrics.json"
    ):
        # Fallback to baseline files if general files are not found
        if not os.path.exists(base_svm_path):
            base_svm_path = "final_svm_model.pkl"
        if not os.path.exists(word_vec_path):
            word_vec_path = "final_word_vectorizer.pkl"
        if not os.path.exists(char_vec_path):
            char_vec_path = "final_char_vectorizer.pkl"

        self.base_model = joblib.load(base_svm_path)
        self.calibrated_model = joblib.load(calibrated_clf_path) if os.path.exists(calibrated_clf_path) else None
        self.word_vec = joblib.load(word_vec_path)
        self.char_vec = joblib.load(char_vec_path)

        # Precompute vocabulary and weights for explanations
        self.word_feature_names = np.array(self.word_vec.get_feature_names_out())
        self.char_feature_names = np.array(self.char_vec.get_feature_names_out())

        coef = self.base_model.coef_.ravel()
        n_words = len(self.word_feature_names)
        self.word_weights = coef[:n_words]
        self.char_weights = coef[n_words:]
        self.intercept = float(self.base_model.intercept_[0]) if hasattr(self.base_model.intercept_, '__len__') else float(self.base_model.intercept_)

        # Load optional domain-specialized models (e.g. personal statements)
        self.ps_base_model = joblib.load("personal_statement_linear_svm.pkl") if os.path.exists("personal_statement_linear_svm.pkl") else None
        self.ps_calibrated = joblib.load("personal_statement_calibrated_clf.pkl") if os.path.exists("personal_statement_calibrated_clf.pkl") else None
        self.ps_word_vec = joblib.load("personal_statement_word_vectorizer.pkl") if os.path.exists("personal_statement_word_vectorizer.pkl") else None
        self.ps_char_vec = joblib.load("personal_statement_char_vectorizer.pkl") if os.path.exists("personal_statement_char_vectorizer.pkl") else None

        if self.ps_base_model is not None and self.ps_word_vec is not None and self.ps_char_vec is not None:
            self.ps_word_feature_names = np.array(self.ps_word_vec.get_feature_names_out())
            self.ps_char_feature_names = np.array(self.ps_char_vec.get_feature_names_out())
            ps_coef = self.ps_base_model.coef_.ravel()
            ps_n_words = len(self.ps_word_feature_names)
            self.ps_word_weights = ps_coef[:ps_n_words]
            self.ps_char_weights = ps_coef[ps_n_words:]
            self.ps_intercept = float(self.ps_base_model.intercept_[0]) if hasattr(self.ps_base_model.intercept_, '__len__') else float(self.ps_base_model.intercept_)
        else:
            self.ps_word_feature_names = None
            self.ps_word_weights = None
            self.ps_intercept = 0.0

        # Load domain catalog and metrics
        self.domain_catalog = {}
        if os.path.exists(catalog_path):
            with open(catalog_path, "r") as f:
                self.domain_catalog = json.load(f)

        self.metrics_info = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics_info = json.load(f)

    def classify_and_explain(self, text: str, domain: str = "general_text", top_k: int = 8) -> Dict[str, Any]:
        """
        Classifies input text and extracts feature contributions.
        Returns explicit domain status and uncertainty caveats.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "error": "Empty text provided",
                "label": "Unknown",
                "label_code": -1,
                "decision_score": 0.0,
                "decision_margin": 0.0,
                "calibrated_probability": None,
                "confidence_label": "Neutral",
                "domain": domain,
                "domain_status": "Unknown",
                "domain_caveat": "No text provided for analysis.",
                "top_deceptive_features": [],
                "top_truthful_features": [],
                "word_highlights": []
            }

        # Select model and vectorizers: use specialized personal statement model when selected
        if domain == "personal_statement" and self.ps_base_model is not None and self.ps_word_vec is not None and self.ps_char_vec is not None:
            active_word_vec = self.ps_word_vec
            active_char_vec = self.ps_char_vec
            active_base_model = self.ps_base_model
            active_calibrated_model = self.ps_calibrated
            active_word_feature_names = self.ps_word_feature_names
            active_word_weights = self.ps_word_weights
            active_intercept = self.ps_intercept
        else:
            active_word_vec = self.word_vec
            active_char_vec = self.char_vec
            active_base_model = self.base_model
            active_calibrated_model = self.calibrated_model
            active_word_feature_names = self.word_feature_names
            active_word_weights = self.word_weights
            active_intercept = self.intercept

        # Vectorize
        x_w = active_word_vec.transform([cleaned])
        x_c = active_char_vec.transform([cleaned])
        x_comb = hstack([x_w, x_c]).tocsr()

        # Decision score
        decision_score = float(active_base_model.decision_function(x_comb)[0])
        pred_code = int(active_base_model.predict(x_comb)[0])
        label = "Deceptive" if pred_code == 1 else "Truthful"

        # Calibrated probability
        calibrated_prob = None
        if active_calibrated_model is not None:
            calibrated_prob = float(active_calibrated_model.predict_proba(x_comb)[0][1])

        margin = abs(decision_score)
        if margin < 0.3:
            conf_label = "Low Margin (Borderline)"
        elif margin < 0.8:
            conf_label = "Moderate Margin"
        elif margin < 1.5:
            conf_label = "High Margin"
        else:
            conf_label = "Very High Margin"

        # Domain validation status and caveat
        domain_meta = self.domain_catalog.get(domain, {})
        val_status = domain_meta.get("validation_status", "Supported / General text transfer")
        is_validated = "Validated" in val_status

        if is_validated:
            if domain == "hotel_review":
                caveat = (
                    f"Model evaluated on {domain_meta.get('display_name', domain)} with verified 88.75% test accuracy "
                    "(Ott et al. gold standard). Predictions reflect resemblance to benchmark deception markers."
                )
            elif domain == "personal_statement":
                caveat = (
                    f"Model evaluated on {domain_meta.get('display_name', domain)} with verified 62.55% test accuracy "
                    "across 103 held-out contributors (Pérez-Rosas & Mihalcea 2015). Notice: First-person statements exhibit "
                    "subtle deception cues; output indicates statistical resemblance to research statements, not universal deception detection."
                )
            else:
                caveat = (
                    f"Model evaluated on {domain_meta.get('display_name', domain)} ({val_status}). "
                    "Predictions reflect resemblance to benchmark training data."
                )
        else:
            caveat = (
                f"Notice: The selected domain ({domain_meta.get('display_name', domain)}) utilizes general lexical "
                "and n-gram transfer. Domain-specific ground truth has not yet been validated. "
                "Treat output strictly as an exploratory hypothesis."
            )

        # Feature contributions: w_j * x_j
        w_indices = x_w.nonzero()[1]
        w_values = x_w.data
        contributions = []
        for idx, val in zip(w_indices, w_values):
            feat = active_word_feature_names[idx]
            weight = active_word_weights[idx]
            contrib = float(weight * val)
            contributions.append({
                "feature": feat,
                "weight": float(weight),
                "tfidf": float(val),
                "contribution": contrib
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_deceptive = [c for c in contributions if c["contribution"] > 0][:top_k]
        top_truthful = [c for c in reversed(contributions) if c["contribution"] < 0][:top_k]

        # Token highlights
        token_map = {c["feature"]: c["contribution"] for c in contributions}
        tokens = cleaned.split()
        word_highlights = []
        for token in tokens:
            impact = token_map.get(token, 0.0)
            if impact > 0.01:
                cat = "deceptive"
            elif impact < -0.01:
                cat = "truthful"
            else:
                cat = "neutral"
            word_highlights.append({
                "word": token,
                "impact": round(float(impact), 4),
                "category": cat
            })

        return {
            "cleaned_text": cleaned,
            "label": label,
            "label_code": pred_code,
            "decision_score": round(decision_score, 4),
            "decision_margin": round(margin, 4),
            "calibrated_probability": round(calibrated_prob, 4) if calibrated_prob is not None else None,
            "confidence_label": conf_label,
            "domain": domain,
            "domain_display": domain_meta.get("display_name", domain.replace("_", " ").title()),
            "domain_status": val_status,
            "is_domain_validated": is_validated,
            "domain_caveat": caveat,
            "intercept": round(active_intercept, 4),
            "top_deceptive_features": top_deceptive,
            "top_truthful_features": top_truthful,
            "word_highlights": word_highlights,
            "scientific_disclaimer": (
                "General text classification evaluates statistical resemblance to labeled datasets. "
                "It does NOT establish factual veracity or personal intent to deceive."
            )
        }

if __name__ == "__main__":
    clf = GeneralClassifier()
    sample = "We stayed at this hotel during our vacation. The bathroom had mold and the AC was broken."
    res = clf.classify_and_explain(sample, domain="hotel_review")
    print("Label:", res["label"])
    print("Decision Score:", res["decision_score"])
    print("Calibrated Probability:", res["calibrated_probability"])
    print("Domain Caveat:", res["domain_caveat"])
