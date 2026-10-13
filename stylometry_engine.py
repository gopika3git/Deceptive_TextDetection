"""
stylometry_engine.py
Engine B — Personal Historical Writing Stylometry & Comparison Engine.

Quantifies individual writing style patterns across linguistic, syntactic,
and lexical dimensions without inferring dishonesty or psychological intent.
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import textstat

# Standard English function words list for stylometric profiling
FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"}

FEATURE_METADATA = {
    "avg_sentence_length": {
        "label": "Avg Sentence Length (words)",
        "description": "Mean number of words per sentence",
        "category": "Structure"
    },
    "avg_word_length": {
        "label": "Avg Word Length (chars)",
        "description": "Mean character length per word",
        "category": "Vocabulary"
    },
    "vocabulary_diversity_ttr": {
        "label": "Vocabulary Diversity (TTR)",
        "description": "Type-Token Ratio (unique words divided by total words)",
        "category": "Vocabulary"
    },
    "hapax_legomena_ratio": {
        "label": "Rare Word Ratio (Hapax)",
        "description": "Proportion of words used exactly once",
        "category": "Vocabulary"
    },
    "function_word_ratio": {
        "label": "Function Word Usage (%)",
        "description": "Proportion of structural filler and function words",
        "category": "Style"
    },
    "first_person_pronouns": {
        "label": "1st Person Pronouns (%)",
        "description": "Self-referential pronouns (I, me, my, we, our)",
        "category": "Pronouns"
    },
    "second_person_pronouns": {
        "label": "2nd Person Pronouns (%)",
        "description": "Addressee-directed pronouns (you, your)",
        "category": "Pronouns"
    },
    "third_person_pronouns": {
        "label": "3rd Person Pronouns (%)",
        "description": "External reference pronouns (he, she, they, it)",
        "category": "Pronouns"
    },
    "comma_frequency": {
        "label": "Commas per 100 words",
        "description": "Frequency of comma punctuation",
        "category": "Punctuation"
    },
    "period_frequency": {
        "label": "Periods per 100 words",
        "description": "Frequency of terminal period punctuation",
        "category": "Punctuation"
    },
    "exclamation_frequency": {
        "label": "Exclamations per 100 words",
        "description": "Frequency of exclamation marks",
        "category": "Punctuation"
    },
    "question_frequency": {
        "label": "Questions per 100 words",
        "description": "Frequency of question marks",
        "category": "Punctuation"
    },
    "semicolon_colon_frequency": {
        "label": "Colons/Semicolons per 100 words",
        "description": "Frequency of structural colons and semicolons",
        "category": "Punctuation"
    },
    "uppercase_word_ratio": {
        "label": "Uppercase Words (%)",
        "description": "Proportion of words typed in ALL CAPS",
        "category": "Style"
    },
    "contraction_ratio": {
        "label": "Contractions (%)",
        "description": "Frequency of contracted words (don't, it's, etc.)",
        "category": "Style"
    },
    "flesch_reading_ease": {
        "label": "Flesch Reading Ease",
        "description": "Comprehension ease index (0-100, higher is easier)",
        "category": "Readability"
    },
    "flesch_kincaid_grade": {
        "label": "Flesch-Kincaid Grade Level",
        "description": "US school grade reading level equivalent",
        "category": "Readability"
    }
}


class StylometricFeatureExtractor:
    """Extracts fine-grained quantitative stylometric features from input text."""

    @staticmethod
    def extract(text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {feat: 0.0 for feat in FEATURE_METADATA.keys()} | {
                "word_count": 0,
                "sentence_count": 0,
                "char_count": 0
            }

        raw = text.strip()
        char_count = len(raw)

        # Paragraphs & Sentences
        paragraphs = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
        paragraph_count = max(1, len(paragraphs))

        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
        sentences = [s.strip() for s in re.split(sentence_pattern, raw) if s.strip()]
        if not sentences:
            sentences = [raw]
        sentence_count = max(1, len(sentences))

        # Words
        words = re.findall(r"\b[A-Za-z0-9'-]+\b", raw)
        word_count = len(words)
        if word_count == 0:
            return {feat: 0.0 for feat in FEATURE_METADATA.keys()} | {
                "word_count": 0,
                "sentence_count": sentence_count,
                "char_count": char_count
            }

        words_lower = [w.lower() for w in words]

        # 1. Structural lengths
        avg_sentence_len = word_count / sentence_count
        avg_word_len = sum(len(w) for w in words) / word_count
        avg_paragraph_len = sentence_count / paragraph_count

        # 2. Vocabulary Richness
        unique_words = set(words_lower)
        ttr = len(unique_words) / word_count

        word_freqs = {}
        for w in words_lower:
            word_freqs[w] = word_freqs.get(w, 0) + 1
        hapax_count = sum(1 for count in word_freqs.values() if count == 1)
        hapax_ratio = hapax_count / word_count

        # 3. Pronoun distributions
        first_person = sum(1 for w in words_lower if w in FIRST_PERSON_PRONOUNS) / word_count * 100.0
        second_person = sum(1 for w in words_lower if w in SECOND_PERSON_PRONOUNS) / word_count * 100.0
        third_person = sum(1 for w in words_lower if w in THIRD_PERSON_PRONOUNS) / word_count * 100.0

        # 4. Function word usage
        func_count = sum(1 for w in words_lower if w in FUNCTION_WORDS)
        func_ratio = (func_count / word_count) * 100.0

        # 5. Punctuation frequencies (per 100 words)
        scale = 100.0 / word_count
        comma_freq = len(re.findall(r",", raw)) * scale
        period_freq = len(re.findall(r"\.", raw)) * scale
        exclamation_freq = len(re.findall(r"!", raw)) * scale
        question_freq = len(re.findall(r"\?", raw)) * scale
        semi_colon_freq = len(re.findall(r"[:;]", raw)) * scale
        ellipsis_dash_freq = len(re.findall(r"(\.\.\.|—|--|-)", raw)) * scale
        quote_freq = len(re.findall(r"[\"']", raw)) * scale

        # 6. Stylistic capitalization and contractions
        caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
        caps_ratio = (caps_words / word_count) * 100.0

        contractions = sum(1 for w in words if "'" in w or "’" in w)
        contraction_ratio = (contractions / word_count) * 100.0

        # 7. Readability
        try:
            flesch_ease = float(textstat.flesch_reading_ease(raw))
            fk_grade = float(textstat.flesch_kincaid_grade(raw))
        except Exception:
            flesch_ease = 60.0
            fk_grade = 8.0

        return {
            "avg_sentence_length": round(avg_sentence_len, 2),
            "avg_word_length": round(avg_word_len, 2),
            "vocabulary_diversity_ttr": round(ttr, 4),
            "hapax_legomena_ratio": round(hapax_ratio, 4),
            "function_word_ratio": round(func_ratio, 2),
            "first_person_pronouns": round(first_person, 2),
            "second_person_pronouns": round(second_person, 2),
            "third_person_pronouns": round(third_person, 2),
            "comma_frequency": round(comma_freq, 2),
            "period_frequency": round(period_freq, 2),
            "exclamation_frequency": round(exclamation_freq, 2),
            "question_frequency": round(question_freq, 2),
            "semicolon_colon_frequency": round(semi_colon_freq, 2),
            "uppercase_word_ratio": round(caps_ratio, 2),
            "contraction_ratio": round(contraction_ratio, 2),
            "flesch_reading_ease": round(flesch_ease, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            # Metadata counts
            "word_count": word_count,
            "sentence_count": sentence_count,
            "char_count": char_count,
            "paragraph_count": paragraph_count
        }


class BaselineAggregator:
    """Aggregates multiple historical samples into a statistical baseline profile."""

    @staticmethod
    def compute_baseline(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes mean and std for each feature across historical samples.
        Evaluates data sufficiency and reliability.
        """
        # Filter for representative samples
        rep_samples = [s for s in samples if s.get("is_representative", True)]
        sample_count = len(rep_samples)

        if sample_count == 0:
            return {
                "sample_count": 0,
                "total_word_count": 0,
                "reliability_rating": "Insufficient Data",
                "reliability_badge": "danger",
                "is_operationally_ready": False,
                "readiness_status": "No Samples Added",
                "readiness_badge": "danger",
                "min_samples_needed": 3,
                "min_words_needed": 200,
                "readiness_detail": "Baseline has 0 samples. Add at least 3 historical samples and 200 words to establish initial operational variance.",
                "scientific_limitation_note": "Operational readiness indicates the statistical algorithm can compute variance across features. It does not guarantee universal scientific validity across different contexts.",
                "sufficiency_warnings": ["No representative historical samples found in profile."],
                "feature_stats": {},
                "timeline": []
            }

        # Extract features for all samples
        extracted = []
        timeline = []
        for s in rep_samples:
            feats = StylometricFeatureExtractor.extract(s["text"])
            feats["sample_id"] = s.get("id")
            feats["date"] = s.get("date", "Unknown")
            feats["context"] = s.get("context", "general")
            extracted.append(feats)
            timeline.append({
                "sample_id": s.get("id"),
                "date": s.get("date", "Unknown"),
                "context": s.get("context", "general"),
                "word_count": feats["word_count"],
                "avg_sentence_length": feats["avg_sentence_length"],
                "ttr": feats["vocabulary_diversity_ttr"],
                "flesch_grade": feats["flesch_kincaid_grade"]
            })

        total_words = sum(f["word_count"] for f in extracted)

        # Compute mean and standard deviation per feature
        feature_stats = {}
        for feat_key, meta in FEATURE_METADATA.items():
            vals = [f[feat_key] for f in extracted]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            # Minimum variance threshold to avoid division by zero
            min_std = max(0.01, mean_val * 0.05) if mean_val > 0 else 0.05
            std_val = max(std_val, min_std)

            feature_stats[feat_key] = {
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "min": round(float(np.min(vals)), 3),
                "max": round(float(np.max(vals)), 3),
                "label": meta["label"],
                "description": meta["description"],
                "category": meta["category"]
            }

        # Evaluate reliability
        warnings = []
        if sample_count < 3:
            warnings.append(f"Small sample count ({sample_count} sample(s)). Minimum recommended is 3-5 samples.")
        if total_words < 200:
            warnings.append(f"Low total word volume ({total_words} words). Minimum recommended is 200-500 words.")

        if sample_count >= 5 and total_words >= 500:
            reliability = "High (Robust Baseline)"
            badge = "success"
        elif sample_count >= 3 and total_words >= 200:
            reliability = "Moderate"
            badge = "warning"
        else:
            reliability = "Low / Insufficient Data"
            badge = "danger"

        # Operational readiness assessment
        is_operationally_ready = (sample_count >= 3 and total_words >= 200)
        min_samples_needed = max(0, 3 - sample_count)
        min_words_needed = max(0, 200 - total_words)

        if is_operationally_ready:
            readiness_status = "Operationally Ready for Comparison"
            readiness_badge = "success"
            readiness_detail = (
                f"Baseline meets minimum operational criteria ({sample_count} samples, {total_words} words). "
                "The engine can compute baseline statistics (mean and standard deviation) across 17 stylometric dimensions for personal comparison."
            )
        else:
            readiness_status = "Needs More Writing Samples"
            readiness_badge = "warning"
            readiness_detail = (
                f"Baseline currently has {sample_count} sample(s) and {total_words} words. "
                f"Add at least {min_samples_needed} more sample(s) and {min_words_needed} more words to establish operational baseline variance."
            )

        scientific_limitation_note = (
            "Operational Readiness vs Scientific Validity Notice: Meeting the minimum operational threshold allows the engine to calculate mathematical deviations (Z-scores). "
            "However, operational readiness does NOT constitute proof of universal scientific reliability. Human writing style varies across communication channels, audience, urgency, and emotional states. "
            "Continual additions of diverse, authentic writing samples improve baseline robustness."
        )

        return {
            "sample_count": sample_count,
            "total_word_count": total_words,
            "reliability_rating": reliability,
            "reliability_badge": badge,
            "is_operationally_ready": is_operationally_ready,
            "readiness_status": readiness_status,
            "readiness_badge": readiness_badge,
            "readiness_detail": readiness_detail,
            "scientific_limitation_note": scientific_limitation_note,
            "min_samples_needed": min_samples_needed,
            "min_words_needed": min_words_needed,
            "sufficiency_warnings": warnings,
            "feature_stats": feature_stats,
            "timeline": timeline
        }


class StylometricComparator:
    """Compares current text against an author's personal historical baseline."""

    @staticmethod
    def compare(current_text: str, baseline_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares new text with the personal baseline.
        Computes Z-scores, deviation classifications, and stylometric alignment index.
        """
        current_feats = StylometricFeatureExtractor.extract(current_text)
        word_count = current_feats["word_count"]

        if word_count == 0:
            return {
                "error": "Empty text provided for comparison",
                "alignment_index": 0.0,
                "shifted_features": [],
                "stable_features": []
            }

        feature_stats = baseline_profile.get("feature_stats", {})
        if not feature_stats:
            return {
                "error": "Baseline profile has no statistical features. Add historical samples first.",
                "alignment_index": 0.0,
                "shifted_features": [],
                "stable_features": []
            }

        shifted_features = []
        stable_features = []
        all_feature_comparisons = []
        z_scores = []

        for feat_key, meta in FEATURE_METADATA.items():
            if feat_key not in feature_stats:
                continue

            stat = feature_stats[feat_key]
            mean = stat["mean"]
            std = stat["std"]
            val = current_feats[feat_key]

            # Calculate Z-score
            z = (val - mean) / std if std > 0 else 0.0
            z_scores.append(abs(z))

            # Relative change percentage
            rel_change = ((val - mean) / mean * 100.0) if mean != 0 else (100.0 if val != 0 else 0.0)

            abs_z = abs(z)
            if abs_z >= 2.0:
                shift_level = "Substantial Shift"
                shift_severity = "high"
            elif abs_z >= 1.0:
                shift_level = "Moderate Shift"
                shift_severity = "moderate"
            else:
                shift_level = "Consistent with Baseline"
                shift_severity = "low"

            item = {
                "feature": feat_key,
                "label": meta["label"],
                "category": meta["category"],
                "description": meta["description"],
                "current_value": round(val, 2),
                "baseline_mean": round(mean, 2),
                "baseline_std": round(std, 2),
                "z_score": round(z, 2),
                "relative_change_pct": round(rel_change, 1),
                "shift_level": shift_level,
                "shift_severity": shift_severity
            }

            all_feature_comparisons.append(item)
            if abs_z >= 1.0:
                shifted_features.append(item)
            else:
                stable_features.append(item)

        # Sort shifted features by deviation severity
        shifted_features.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        stable_features.sort(key=lambda x: abs(x["z_score"]))

        # Stylometric Alignment Index (0% to 100%)
        # Alignment = 100% * max(0, 1 - mean(min(1, |Z| / 3.0)))
        if z_scores:
            penalty = np.mean([min(1.0, z / 3.0) for z in z_scores])
            alignment_index = max(0.0, round(float(100.0 * (1.0 - penalty)), 1))
        else:
            alignment_index = 100.0

        # Qualitative synthesis
        if alignment_index >= 80.0:
            overall_assessment = "Strong Stylistic Consistency"
            assessment_color = "success"
            summary_statement = "The text closely matches the author's historical writing baseline across sentence length, punctuation, and lexical habits."
        elif alignment_index >= 55.0:
            overall_assessment = "Moderate Stylistic Divergence"
            assessment_color = "warning"
            summary_statement = f"The text exhibits noticeable variations in {len(shifted_features)} stylistic features compared to the baseline."
        else:
            overall_assessment = "Substantial Stylistic Deviation"
            assessment_color = "danger"
            summary_statement = "The text diverges markedly from the author's historical writing habits across multiple dimensions."

        # Alternative benign explanations
        alternative_explanations = [
            "Differing Communication Channel: Writing on mobile devices, chat platforms, or constrained forms typically causes shorter sentences and altered punctuation.",
            "Formality & Audience Shift: Shifting from informal to professional contexts (or vice versa) markedly alters vocabulary diversity and pronoun usage.",
            "Time Pressure & Urgency: Rushed or draft composition often reduces punctuation usage and sentence complexity.",
            "Collaborative Writing & Assistive Tools: Use of templates, AI writing assistants, or editorial feedback creates substantial stylistic shifts.",
            "Topic & Technicality: Highly technical or specialized subjects naturally shift word length, readability scores, and vocabulary richness."
        ]

        return {
            "alignment_index": alignment_index,
            "overall_assessment": overall_assessment,
            "assessment_color": assessment_color,
            "summary_statement": summary_statement,
            "current_text_stats": {
                "word_count": current_feats["word_count"],
                "sentence_count": current_feats["sentence_count"]
            },
            "baseline_summary": {
                "sample_count": baseline_profile.get("sample_count", 0),
                "total_words": baseline_profile.get("total_word_count", 0),
                "reliability_rating": baseline_profile.get("reliability_rating", "Unknown")
            },
            "shifted_features_count": len(shifted_features),
            "stable_features_count": len(stable_features),
            "top_shifted_features": shifted_features[:5],
            "stable_features": stable_features[:5],
            "all_features": all_feature_comparisons,
            "alternative_explanations": alternative_explanations,
            "scientific_disclaimer": (
                "Scientific Caution: Stylometric divergence reflects measurable changes in writing habits, "
                "not dishonesty. Style shifts frequently occur due to audience, medium, topic, fatigue, or mood. "
                "These metrics must NOT be interpreted as proof of deception."
            )
        }

if __name__ == "__main__":
    sample_text1 = "Hello team, I wanted to provide a quick update on our project progress. We have completed the initial data audit."
    sample_text2 = "Further to our meeting, I am writing to confirm the delivery schedule for next week. Please let me know if you need adjustments."
    sample_text3 = "Thank you for the prompt response. We reviewed the specifications and believe we can achieve the target milestones."
    
    samples = [
        {"id": "1", "text": sample_text1, "date": "2026-09-01", "context": "email", "is_representative": True},
        {"id": "2", "text": sample_text2, "date": "2026-09-15", "context": "email", "is_representative": True},
        {"id": "3", "text": sample_text3, "date": "2026-10-01", "context": "email", "is_representative": True}
    ]
    
    baseline = BaselineAggregator.compute_baseline(samples)
    print("Baseline Reliability:", baseline["reliability_rating"])
    
    test_current = "HEY MAN!! WHERE ARE U?? COME QUICK ASAP!! NEED HELP!!!"
    comp = StylometricComparator.compare(test_current, baseline)
    print("Alignment Index:", comp["alignment_index"])
    print("Assessment:", comp["overall_assessment"])
    print("Top shifted:", [f["label"] for f in comp["top_shifted_features"]])
