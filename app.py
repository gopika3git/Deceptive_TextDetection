"""
app.py
General Text Deception Detection & Personal Historical Writing Analysis System.
FastAPI Backend providing Dual Analysis Engines:
- Engine A: General Text Classifier (Multi-domain TF-IDF Linear SVM with calibrated probability)
- Engine B: Personal Historical Writing Stylometry & Baseline Comparison
- Combined Interpretation: Side-by-side synthesis with ethical disclaimers
- Profile & Sample Management: Local storage, privacy controls, and sample deletion
"""

import os
import json
import csv
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from general_classifier import GeneralClassifier, clean_text
from stylometry_engine import StylometricComparator
from profile_manager import ProfileManager
from combined_analyzer import CombinedAnalyzer

# Initialize App
app = FastAPI(
    title="General Text Deception Detection & Personal Stylometry API",
    description="Dual-engine system combining general text classification with personal historical writing stylometry.",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engines
classifier = GeneralClassifier()
profile_mgr = ProfileManager()
combined_analyzer = CombinedAnalyzer(classifier=classifier, profile_manager=profile_mgr)

# Feedback file
FEEDBACK_FILE = "feedback_log.csv"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "domain", "predicted_label", "user_label", "comments"])


# ===================== PYDANTIC SCHEMAS =====================

class GeneralPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Raw text to classify")
    domain: Optional[str] = Field(default="general_text", description="Target text domain")
    top_k: Optional[int] = Field(default=8, ge=1, le=50, description="Top explanatory features count")

class FeedbackRequest(BaseModel):
    review_text: Optional[str] = Field(default="", max_length=2000, description="Review text or sample snippet")
    domain: Optional[str] = Field(default="general_text", description="Text domain")
    predicted_label: str = Field(..., description="The predicted label ('Truthful' or 'Deceptive')")
    user_label: str = Field(..., description="User perceived label ('Truthful' or 'Deceptive')")
    comments: Optional[str] = Field(default="", max_length=500, description="Optional user comments")

class SampleAddRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=50000, description="Historical genuine writing sample text from the target author")
    date: Optional[str] = Field(default=None, description="Approximate composition date (YYYY-MM-DD)")
    context: Optional[str] = Field(default="professional_email", description="Writing channel/context")
    source_description: Optional[str] = Field(default="", max_length=200, description="Sample title or verified source")
    is_representative: Optional[bool] = Field(default=True, description="Whether sample represents authentic writing from this author")

class ProfileCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Target author or subject name")
    description: Optional[str] = Field(default="", max_length=250, description="Role, subject context, or case identifier")
    notes: Optional[str] = Field(default="", max_length=500, description="Investigative notes or provenance of verified baseline samples")
    user_authorization_confirmed: bool = Field(
        ...,
        description="Confirmation that investigator is authorized to store and analyze writing samples for this target subject"
    )
    initial_samples: Optional[List[SampleAddRequest]] = Field(
        default=None,
        description="Optional initial historical writing samples from this same target author"
    )

class ComparePersonalRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=50000, description="Submitted text to compare against target author baseline")
    profile_id: str = Field(..., description="Target author profile ID")

class CombinedAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=50000, description="Submitted text to analyze")
    domain: Optional[str] = Field(default="general_text", description="Target text domain")
    profile_id: Optional[str] = Field(default=None, description="Optional target author profile ID for Engine B stylometric comparison")


# ===================== HEALTH & DOMAIN ENDPOINTS =====================

@app.get("/health")
def health_check():
    """System health check and engine readiness."""
    is_model_loaded = classifier.base_model is not None
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": is_model_loaded,
        "engine_a_loaded": is_model_loaded,
        "engine_b_profiles_count": len(profile_mgr.list_profiles())
    }

@app.get("/api/domains")
def get_domains():
    """Returns catalog of supported domains and their empirical validation status."""
    return {
        "status": "success",
        "catalog": classifier.domain_catalog
    }


# ===================== ENGINE A: GENERAL CLASSIFIER =====================

@app.post("/predict")
def legacy_predict(req: GeneralPredictRequest):
    """
    Backwards-compatible prediction endpoint.
    Classifies text as Truthful or Deceptive with exact Linear SVM decision score and explanations.
    """
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = classifier.classify_and_explain(raw_text, domain=req.domain, top_k=req.top_k)
    return {
        "status": "success",
        "label": result["label"],
        "label_code": result["label_code"],
        "decision_score": result["decision_score"],
        "decision_margin": result["decision_margin"],
        "calibrated_probability": result["calibrated_probability"],
        "confidence_label": result["confidence_label"],
        "domain": result["domain"],
        "domain_display": result["domain_display"],
        "domain_status": result["domain_status"],
        "domain_caveat": result["domain_caveat"],
        "top_deceptive_features": result["top_deceptive_features"],
        "top_truthful_features": result["top_truthful_features"],
        "word_highlights": result["word_highlights"],
        "explanation_note": "Decision score represents signed distance from SVM hyperplane. Positive pushes toward Deceptive; negative toward Truthful."
    }

@app.post("/explain")
def legacy_explain(req: GeneralPredictRequest):
    """Detailed feature contribution breakdown for model interpretability."""
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = classifier.classify_and_explain(raw_text, domain=req.domain, top_k=req.top_k)
    return {
        "status": "success",
        **result
    }


# ===================== ENGINE B: PROFILE & STYLOMETRY =====================

@app.get("/api/profiles")
def list_profiles():
    """List summary info for all stored personal writing profiles."""
    profiles = profile_mgr.list_profiles()
    return {
        "status": "success",
        "profiles": profiles
    }

@app.post("/api/profiles")
def create_profile(req: ProfileCreateRequest):
    """Create a new target author profile with authorization confirmation."""
    if not req.user_authorization_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Authorization confirmation is required to create a target author profile."
        )
    samples_data = None
    if req.initial_samples:
        samples_data = [s.model_dump() for s in req.initial_samples]

    profile = profile_mgr.create_profile(
        name=req.name.strip(),
        description=req.description.strip(),
        notes=req.notes.strip(),
        user_authorization_confirmed=req.user_authorization_confirmed,
        initial_samples=samples_data
    )
    return {
        "status": "success",
        "message": f"Target author profile '{profile['name']}' created successfully.",
        "profile": profile
    }

@app.get("/api/profiles/{profile_id}")
def get_profile(profile_id: str):
    """Retrieve full target author profile details, samples, and computed baseline statistics."""
    profile = profile_mgr.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Target author profile not found.")
    return {
        "status": "success",
        "profile": profile
    }

@app.get("/api/profiles/{profile_id}/readiness")
def get_profile_readiness(profile_id: str):
    """Evaluates baseline data sufficiency and operational readiness for a target author profile."""
    profile = profile_mgr.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Target author profile not found.")

    baseline = profile.get("baseline", {})
    sample_count = len(profile.get("samples", []))
    total_words = baseline.get("total_word_count", 0)

    is_ready = bool(baseline.get("is_operationally_ready", sample_count >= 3 and total_words >= 200))
    return {
        "status": "success",
        "profile_id": profile["id"],
        "profile_name": profile["name"],
        "is_demo": profile.get("is_demo", False),
        "sample_count": sample_count,
        "total_words": total_words,
        "is_operationally_ready": is_ready,
        "readiness_status": baseline.get(
            "readiness_status",
            "Operationally Ready for Comparison" if is_ready else "Needs More Writing Samples"
        ),
        "readiness_badge": baseline.get("readiness_badge", "success" if is_ready else "warning"),
        "min_samples_needed": baseline.get("min_samples_needed", max(0, 3 - sample_count)),
        "min_words_needed": baseline.get("min_words_needed", max(0, 200 - total_words)),
        "readiness_detail": baseline.get("readiness_detail", ""),
        "scientific_limitation_note": baseline.get(
            "scientific_limitation_note",
            "Operational readiness indicates the statistical algorithm can compute variance across features. It does not guarantee universal scientific validity."
        ),
        "reliability_rating": baseline.get("reliability_rating", "Unknown"),
        "reliability_badge": baseline.get("reliability_badge", "secondary")
    }

@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    """Permanently delete profile and purge all stored sample text (Privacy compliance)."""
    if profile_id == "demo_analyst":
        raise HTTPException(status_code=403, detail="The built-in demonstration profile 'demo_analyst' cannot be deleted.")
    success = profile_mgr.delete_profile(profile_id)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return {
        "status": "success",
        "message": "Profile and all associated sample text permanently purged."
    }

@app.post("/api/profiles/{profile_id}/samples")
def add_sample_to_profile(profile_id: str, req: SampleAddRequest):
    """Add a historical sample and recompute the statistical baseline."""
    profile = profile_mgr.add_sample(
        profile_id=profile_id,
        text=req.text,
        date=req.date,
        context=req.context,
        source_description=req.source_description,
        is_representative=req.is_representative
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return {
        "status": "success",
        "message": "Sample added and baseline recomputed.",
        "profile": profile
    }

@app.delete("/api/profiles/{profile_id}/samples/{sample_id}")
def delete_sample_from_profile(profile_id: str, sample_id: str):
    """Permanently delete an individual sample and update the baseline."""
    profile = profile_mgr.delete_sample(profile_id, sample_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile or sample not found.")
    return {
        "status": "success",
        "message": "Sample deleted and baseline recomputed.",
        "profile": profile
    }

@app.post("/api/compare-personal")
def compare_personal(req: ComparePersonalRequest):
    """Compares current text against a specific personal writing baseline."""
    profile = profile_mgr.get_profile(req.profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    comp = StylometricComparator.compare(req.text, profile.get("baseline", {}))
    if "error" in comp:
        raise HTTPException(status_code=400, detail=comp["error"])

    return {
        "status": "success",
        "profile_id": profile["id"],
        "profile_name": profile["name"],
        "comparison": comp
    }


# ===================== COMBINED DUAL ANALYSIS =====================

@app.post("/api/analyze-combined")
def analyze_combined(req: CombinedAnalysisRequest):
    """
    Runs both Engine A and Engine B side-by-side.
    Returns separated findings and an ethical combined narrative synthesis.
    """
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    report = combined_analyzer.analyze(
        text=raw_text,
        domain=req.domain,
        profile_id=req.profile_id
    )
    return {
        "status": "success",
        **report
    }


# ===================== METRICS & FEEDBACK =====================

@app.get("/metrics")
def get_metrics():
    """
    Returns general model evaluation metrics and domain-specific validation logs.
    """
    metrics_path = "general_model_metrics.json"
    multi_path = "multi_domain_experiments_metrics.json"
    response_payload = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            response_payload["metrics"] = json.load(f)
    if os.path.exists(multi_path):
        with open(multi_path, "r", encoding="utf-8") as f:
            response_payload["multi_domain_experiments"] = json.load(f)

    if response_payload:
        return {
            "status": "success",
            **response_payload
        }

    # Fallback to model_metadata.json
    legacy_meta = "model_metadata.json"
    if os.path.exists(legacy_meta):
        with open(legacy_meta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "success",
            "metadata": data
        }

    raise HTTPException(status_code=404, detail="Metrics metadata not found.")

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Validates and stores user feedback for model monitoring.
    Avoids logging complete private text if privacy sensitive.
    """
    valid_labels = {"truthful", "deceptive"}
    if req.user_label.strip().lower() not in valid_labels:
        raise HTTPException(status_code=400, detail="User label must be either 'Truthful' or 'Deceptive'.")

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Store snippet up to 100 chars to avoid leaking full private message
            snippet = (req.review_text[:100] + "...") if req.review_text else ""
            writer.writerow([
                timestamp,
                req.domain,
                req.predicted_label.strip(),
                req.user_label.strip().capitalize(),
                req.comments.strip()
            ])
        return {
            "status": "success",
            "message": "Feedback submitted successfully. Thank you for contributing!",
            "timestamp": timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


# ===================== STATIC UI =====================

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the primary UI web application."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Deception Detection API is running. Web UI not loaded.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
