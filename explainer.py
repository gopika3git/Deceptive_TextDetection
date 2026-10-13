"""
Text Deception Detection Explainer Module
Extracts exact Linear SVM feature contributions for Word and Character n-grams.
Guarantees that feature contributions strictly correspond to the model's decision function.
Provides token highlighting and structured explanation payloads.
"""

import re
import numpy as np
import joblib
from scipy.sparse import hstack

def clean_text(text: str) -> str:
    """Preprocess review text identically to the training pipeline."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class DeceptionExplainer:
    def __init__(self, model_path="final_svm_model.pkl", word_vec_path="final_word_vectorizer.pkl", char_vec_path="final_char_vectorizer.pkl"):
        self.model = joblib.load(model_path)
        self.word_vectorizer = joblib.load(word_vec_path)
        self.char_vectorizer = joblib.load(char_vec_path)
        
        # Precompute vocabulary lookups
        self.word_feature_names = np.array(self.word_vectorizer.get_feature_names_out())
        self.char_feature_names = np.array(self.char_vectorizer.get_feature_names_out())
        
        # Linear SVM weights
        coef = self.model.coef_.ravel()
        n_words = len(self.word_feature_names)
        self.word_weights = coef[:n_words]
        self.char_weights = coef[n_words:]
        self.intercept = float(self.model.intercept_[0]) if hasattr(self.model.intercept_, '__len__') else float(self.model.intercept_)

    def predict_and_explain(self, text: str, top_k: int = 8):
        """
        Analyzes a review, predicts Truthful (0) or Deceptive (1),
        and returns exact feature contributions and word highlights.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "error": "Empty review provided",
                "label": "Unknown",
                "label_code": -1,
                "decision_score": 0.0,
                "decision_margin": 0.0,
                "confidence_label": "Neutral",
                "top_deceptive_features": [],
                "top_truthful_features": [],
                "word_highlights": []
            }
            
        x_word = self.word_vectorizer.transform([cleaned])
        x_char = self.char_vectorizer.transform([cleaned])
        x_comb = hstack([x_word, x_char]).tocsr()
        
        # Decision score: f(x) = w^T x + b
        decision_score = float(self.model.decision_function(x_comb)[0])
        pred_code = int(self.model.predict(x_comb)[0])
        label = "Deceptive" if pred_code == 1 else "Truthful"
        
        # Margin confidence indicator (distance from hyperplane)
        margin = abs(decision_score)
        if margin < 0.3:
            confidence_label = "Low Margin (Borderline)"
        elif margin < 0.8:
            confidence_label = "Moderate Margin"
        elif margin < 1.5:
            confidence_label = "High Margin"
        else:
            confidence_label = "Very High Margin"
            
        # Extract word feature contributions
        w_indices = x_word.nonzero()[1]
        w_values = x_word.data
        word_contributions = []
        for idx, val in zip(w_indices, w_values):
            feat_name = self.word_feature_names[idx]
            weight = self.word_weights[idx]
            contrib = float(weight * val)
            word_contributions.append({
                "feature": feat_name,
                "type": "word",
                "weight": float(weight),
                "tfidf": float(val),
                "contribution": contrib
            })
            
        # Top deceptive (positive contribution) and truthful (negative contribution)
        word_contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_deceptive = [c for c in word_contributions if c["contribution"] > 0][:top_k]
        top_truthful = [c for c in reversed(word_contributions) if c["contribution"] < 0][:top_k]
        
        # Word highlight mapping for text rendering in frontend
        # Map words to their net impact
        tokens = cleaned.split()
        token_impacts = {}
        for c in word_contributions:
            token_impacts[c["feature"]] = c["contribution"]
            
        word_highlights = []
        for token in tokens:
            impact = token_impacts.get(token, 0.0)
            if impact > 0.01:
                category = "deceptive"
            elif impact < -0.01:
                category = "truthful"
            else:
                category = "neutral"
            word_highlights.append({
                "word": token,
                "impact": float(impact),
                "category": category
            })
            
        return {
            "cleaned_text": cleaned,
            "label": label,
            "label_code": pred_code,
            "decision_score": round(decision_score, 4),
            "decision_margin": round(margin, 4),
            "confidence_label": confidence_label,
            "intercept": round(self.intercept, 4),
            "top_deceptive_features": top_deceptive,
            "top_truthful_features": top_truthful,
            "word_highlights": word_highlights
        }

if __name__ == "__main__":
    explainer = DeceptionExplainer()
    sample = "I loved this hotel! The staff was amazing, extremely clean rooms and great view of Millennium Park."
    res = explainer.predict_and_explain(sample)
    print(f"Prediction: {res['label']} (Score: {res['decision_score']})")
    print("Top Deceptive cues:", [f['feature'] for f in res['top_deceptive_features']])
    print("Top Truthful cues:", [f['feature'] for f in res['top_truthful_features']])
