"""
combined_analyzer.py
Responsible Synthesis Engine for Dual-Engine Text Deception Analysis.

Integrates:
- Engine A: General Text Classifier (Resemblance to trained deception markers)
- Engine B: Personal Historical Stylometry (Deviation from personal baseline)
Guarantees:
- Strict separation of general classification vs personal stylometrics
- Rejection of pseudo-scientific combined "deception probability" formulas
- Objective, non-judgmental explanations with contextual caveats
"""

from typing import Dict, Any, Optional
from general_classifier import GeneralClassifier
from stylometry_engine import StylometricComparator
from profile_manager import ProfileManager

class CombinedAnalyzer:
    """Orchestrates Engine A and Engine B and produces a responsible side-by-side synthesis."""

    def __init__(self, classifier: Optional[GeneralClassifier] = None, profile_manager: Optional[ProfileManager] = None):
        self.classifier = classifier or GeneralClassifier()
        self.profile_manager = profile_manager or ProfileManager()

    def analyze(
        self,
        text: str,
        domain: str = "general_text",
        profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Engine A and optionally Engine B.
        Generates a transparent combined interpretation.
        """
        # 1. Engine A: General Classifier
        engine_a_result = self.classifier.classify_and_explain(text, domain=domain)

        # 2. Engine B: Personal Stylometric Comparison
        engine_b_result = None
        profile_meta = None
        if profile_id:
            profile = self.profile_manager.get_profile(profile_id)
            if profile:
                profile_meta = {
                    "id": profile["id"],
                    "name": profile["name"],
                    "description": profile.get("description", ""),
                    "sample_count": len(profile.get("samples", [])),
                    "reliability_rating": profile.get("baseline", {}).get("reliability_rating", "Unknown")
                }
                engine_b_result = StylometricComparator.compare(text, profile.get("baseline", {}))
            else:
                engine_b_result = {
                    "error": f"Profile '{profile_id}' not found.",
                    "alignment_index": None
                }

        # 3. Combined Interpretation Synthesis
        combined_synthesis = self._synthesize_findings(engine_a_result, engine_b_result, profile_meta)

        return {
            "engine_a_general_classification": engine_a_result,
            "engine_b_personal_stylometry": engine_b_result,
            "profile_metadata": profile_meta,
            "combined_synthesis": combined_synthesis
        }

    def _synthesize_findings(
        self,
        res_a: Dict[str, Any],
        res_b: Optional[Dict[str, Any]],
        prof: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Creates responsible side-by-side interpretation without merging scores."""
        label_a = res_a.get("label", "Unknown")
        domain_name = res_a.get("domain_display", "General Text")
        is_val = res_a.get("is_domain_validated", False)
        calibrated_prob = res_a.get("calibrated_probability")

        if not res_b:
            summary_statement = (
                f"Engine A classified the text as predicted class '{label_a}' based on general lexical benchmarks "
                f"({domain_name}). No personal historical baseline was provided for this target author; personalized stylometric comparison was not performed."
            )
            stylometry_status = "Not evaluated (No target author baseline supplied)"
            concordance = "Single Engine"
        elif "error" in res_b:
            err_msg = res_b["error"]
            author_name = prof.get("name", "the target author") if prof else "the target author"
            summary_statement = (
                f"Engine A classified the text as predicted class '{label_a}' based on general lexical benchmarks "
                f"({domain_name}). Target author baseline for {author_name} could not be evaluated: {err_msg}"
            )
            stylometry_status = f"Incomplete ({err_msg})"
            concordance = "Single Engine (Baseline Incomplete)"
        else:
            align_idx = res_b.get("alignment_index", 100.0)
            n_shifted = res_b.get("shifted_features_count", 0)
            author_name = prof.get("name", "the target author") if prof else "the target author"
            sample_count = prof.get("sample_count", 0) if prof else 0
            rel_rating = prof.get("reliability_rating", "") if prof else ""
            low_volume = sample_count < 3 or "Low" in rel_rating or "Insufficient" in rel_rating

            if align_idx >= 80.0:
                # Strong Stylistic Consistency
                if label_a == "Deceptive":
                    concordance = "Classifier Divergence / Stylistically Consistent"
                    summary_statement = (
                        f"Dual Analysis Summary: Engine A identified possible deception indicators based on general training benchmarks ({label_a}, "
                        f"calibrated resemblance: {calibrated_prob if calibrated_prob is not None else 'N/A'}), "
                        f"yet Engine B indicates the writing remains strongly consistent with {author_name}'s established personal habits "
                        f"(Alignment Index: {align_idx}%). This suggests vocabulary and phrasing match the target author's normal habits despite general benchmark flags."
                    )
                else:
                    concordance = "Consistent Truthful Alignment"
                    summary_statement = (
                        f"Dual Analysis Summary: The text aligns with truthful training patterns ({label_a}) "
                        f"and closely matches {author_name}'s historical writing baseline across key metrics (Alignment Index: {align_idx}%)."
                    )
            elif align_idx >= 55.0:
                # Moderate Stylistic Divergence
                if label_a == "Deceptive":
                    concordance = "Classifier Deceptive / Moderate Stylistic Divergence"
                    summary_statement = (
                        f"Dual Analysis Summary: Engine A identified possible deception indicators based on general benchmark models "
                        f"({label_a}, calibrated resemblance: {calibrated_prob if calibrated_prob is not None else 'N/A'}), "
                        f"while Engine B observed moderate stylistic divergence from {author_name}'s historical baseline "
                        f"(Alignment Index: {align_idx}%, {n_shifted} shifted features). The writing exhibits noticeable stylistic variation "
                        f"from baseline habits without complete departure. "
                        "Notice: Stylistic shifts often stem from communication medium, urgency, audience, or topic shifts rather than untruthfulness."
                    )
                else:
                    concordance = "Moderate Stylistic Divergence / Classifier Truthful"
                    summary_statement = (
                        f"Dual Analysis Summary: Engine A found lexical cues consistent with truthful benchmark writing ({label_a}), "
                        f"while Engine B observed moderate stylistic divergence from {author_name}'s historical baseline "
                        f"(Alignment Index: {align_idx}%, {n_shifted} shifted features). Although general benchmark markers align with truthful patterns, "
                        f"the writing exhibits noticeable variations in phrasing or structure compared to personal habits. "
                        "Notice: Moderate stylistic variations commonly reflect context, medium, or emotional tone rather than deception."
                    )
            else:
                # Substantial Stylistic Deviation (< 55.0)
                if label_a == "Deceptive":
                    concordance = "Dual Divergence Noted"
                    summary_statement = (
                        f"Dual Analysis Summary: Engine A identified possible deception indicators based on general benchmark models "
                        f"(predicted class: {label_a}, calibrated resemblance: {calibrated_prob if calibrated_prob is not None else 'N/A'}). "
                        f"Simultaneously, Engine B observed that the writing exhibits substantial stylistic deviation from {author_name}'s historical baseline "
                        f"(Alignment Index: {align_idx}%, {n_shifted} shifted features). "
                        "Notice: Stylistic deviations often stem from communication medium, urgency, or topic shifts rather than untruthfulness."
                    )
                else:
                    concordance = "Stylistic Divergence / Classifier Truthful"
                    summary_statement = (
                        f"Dual Analysis Summary: Engine A found lexical cues consistent with truthful benchmark writing ({label_a}), "
                        f"though Engine B noted substantial stylistic deviation from {author_name}'s baseline "
                        f"(Alignment Index: {align_idx}%, {n_shifted} shifted features). "
                        "This pattern reflects marked deviation from established personal writing habits, commonly seen when writing in a different context, format, or emotional state."
                    )

            if low_volume:
                summary_statement += (
                    f" Note: Historical baseline for target author {author_name} is based on limited data "
                    f"({sample_count} sample{'s' if sample_count != 1 else ''}); statistical reliability is low."
                )
                stylometry_status = f"{res_b.get('overall_assessment', 'Evaluated')} ({align_idx}% Alignment - Low Data Volume)"
            else:
                stylometry_status = f"{res_b.get('overall_assessment', 'Evaluated')} ({align_idx}% Alignment)"

        return {
            "concordance_type": concordance,
            "engine_a_headline": f"{label_a} ({domain_name})",
            "engine_b_headline": stylometry_status,
            "synthesis_narrative": summary_statement,
            "domain_validation_note": "Domain Validated" if is_val else "Domain Generalization Transfer (Unvalidated Ground Truth)",
            "methodology_reminder": (
                "Independent Metric Rule: General benchmark predictions and target-author stylometric deviation "
                "measure separate linguistic attributes. They are NOT combined mathematically into an overall deception score."
            )
        }

if __name__ == "__main__":
    analyzer = CombinedAnalyzer()
    sample = "I stayed here last month and was truly disappointed. The room was not cleaned and the noise was intolerable."
    report = analyzer.analyze(sample, domain="hotel_review", profile_id="demo_analyst")
    print("Synthesis Narrative:")
    print(report["combined_synthesis"]["synthesis_narrative"])
