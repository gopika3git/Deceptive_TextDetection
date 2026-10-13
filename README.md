# DSP_Prok — Dual-Engine Text Deception & Stylometric Analysis Platform

**Version:** 3.0.0 (Release Candidate — Verified)  
**System Architecture:** Dual Independent Analysis Engines (Engine A & Engine B) with Non-Blended Responsible Synthesis.

---

## 1. System Overview

DSP_Prok is an investigative text analysis system providing:
- **Engine A (General Classification):** Domain-routed Linear Support Vector Machine (Linear SVM) with Platt Sigmoid Calibration. Compares submitted text against empirical deception corpora (Ott et al. hotel reviews, Pérez-Rosas & Mihalcea personal statements). Outputs signed hyperplane decision margin, calibrated probability ($0.0 - 1.0$), and top informative word/character n-gram feature contributions.
- **Engine B (Personal Stylometry):** 17-dimensional quantitative stylometric comparison against verified personal writing baselines. Computes feature deviations ($Z$-scores) and a Stylometric Alignment Index ($0\% - 100\%$).
- **Combined Analysis:** Side-by-side synthesis keeping general benchmark markers strictly separated from personal stylistic variance. Adheres to the Independent Metric Rule: general classifier outputs and stylometric divergence are never merged mathematically into a pseudoscientific blended "deception score."
- **Privacy & Baseline Governance:** Local profile storage (`profiles_store/`), investigator authorization enforcement, dynamic operational readiness tracking ($\ge 3$ samples, $\ge 200$ words), and instant, permanent sample text purging on deletion.

---

## 2. Dependencies & Supported Python Environment

### Supported Python Versions
- **Python 3.10 to 3.12** (Officially verified and tested on Python 3.12.4 64-bit on Windows).

### Pinned Dependencies (`requirements.txt`)
All production and testing dependencies are strictly pinned to exact versions (`==`) for deterministic, reproducible builds:
- `fastapi==0.140.13` — Web framework and API endpoints
- `uvicorn==0.51.0` — ASGI server
- `pydantic==2.9.2` — Request validation and schema parsing
- `numpy==2.1.1` — Numerical arrays and matrix mathematics
- `scipy==1.18.1` — Statistical computations and distance metrics
- `scikit-learn==1.9.0` — Model loading, SVM inference, and vectorization
- `joblib==1.5.3` — Artifact deserialization
- `pandas==2.2.3` — Dataset management and metrics inspection
- `textstat==0.7.13` — Readability and syllable extraction
- `pytest==9.1.1` — Test suite runner
- `httpx==0.28.1` — Test client HTTP handling
- `websockets==16.1.1` — Async browser automation protocol

---

## 3. Clean Installation Guide

Follow these exact steps to set up the system in a fresh virtual environment:

```bash
# 1. Navigate to the project root directory
cd c:\Users\Pavithran\.gemini\antigravity-ide\scratch\DSP_Prok

# 2. Create a clean Python virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# 4. Install all pinned dependencies
pip install -r requirements.txt
```

---

## 4. Application Startup

Run the development server bound strictly to localhost:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- **Interactive Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **REST API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

> **Important:** Do NOT bind the application to `0.0.0.0` or expose the port to an untrusted local area network or the public internet. See Section 6 for security details.

---

## 5. Verification & Testing

### 1. Automated Test Suite (43 Tests)
```bash
pytest -v
```
Runs 43 automated tests verifying:
- Pipeline artifacts, vectorizers, and zero train/val/test data leakage.
- Hotel review and personal statement domain routing and probability calibration.
- Contributor-grouped split separation ensuring 0% author overlap across held-out sets.
- Target-author profile creation, authorization validation, and baseline readiness thresholds.
- Permanent raw text purging upon sample deletion.
- Demo profile (`demo_analyst`) deletion immutability.
- Path traversal protection and empty sanitized ID rejection.
- All 6 concordance tiers in Combined Dual Analysis.

### 2. Live Browser End-to-End Workflow (9 Steps)
```bash
python test_wizard_browser_workflow.py
```
Executes a 9-step automated browser workflow over native Chrome DevTools Protocol (CDP):
1. **Launch Onboarding Wizard:** Modal opens and Step 1 form displays.
2. **Profile Creation & Authorization:** Authorization checkbox is strictly required; target author created.
3. **Historical Samples Addition:** Samples added with real-time text analysis.
4. **Dynamic Baseline Readiness Progression:** Dynamic transition from "Needs More Writing Samples" to "Operationally Ready" upon exceeding $\ge 3$ samples and $\ge 200$ words.
5. **Engine Selection Synchronization:** Created profile automatically preselected across Engine B and Combined Analysis.
6. **Live Stylometric Evaluation:** Compares submitted text against the newly established target baseline.
7. **Persistence Verification:** Full page reload confirms profiles and baselines persist from disk storage.
8. **Incomplete Baseline Fallback:** Incomplete baselines trigger clear data warnings while allowing general classification.
9. **Demo Profile Isolation:** Built-in demonstration subject (`demo_analyst`) is visually distinct and labeled.

### 3. Security, Privacy & Reproducibility Audit
```bash
python audit_phase3_security_and_readiness.py
```
Validates API boundaries, confirms zero PII retention upon sample deletion, verifies test metrics against locked benchmark evaluation sets, and tests the full Combined Analysis concordance matrix.

---

## 6. Security Boundaries & Deployment Limitations

### ⚠️ Local-Only Workstation Boundary
- **Deployment Status:** **READY FOR LOCAL MVP / DEMO ONLY**.
- **Network / Production Deployment:** **NOT READY FOR NETWORK DEPLOYMENT**.
- **Reason:** The REST API endpoints (`/api/profiles`, `/predict`, `/api/compare-personal`, `/api/analyze-combined`) are **unauthenticated**. Any process or client capable of reaching the HTTP port can create, view, modify, or delete investigator profiles and writing samples.
- **Authorization Checkbox is NOT Authentication:** The UI checkbox (`user_authorization_confirmed: true`) represents the investigator's *self-attestation* that they have authorization to store writing samples. It does **not** authenticate identity, verify cryptographic signatures, or restrict access.
- **Network Deployment Requirement:** Before deploying this software to a shared team network, VPN, or cloud environment, an external authentication reverse proxy or API gateway (e.g., OAuth2-proxy, NGINX with JWT validation, Keycloak) must be deployed in front of the application to enforce user identity and profile-level access control.

### Profile Data Storage & Privacy Guarantees
- Profile records and baseline statistics are stored locally as JSON files in `profiles_store/`.
- Deleting an individual sample immediately removes the raw text from disk storage and recalculates baseline statistics.
- Deleting an entire profile permanently deletes the profile JSON file from disk (`profiles_store/{profile_id}.json`).
- The built-in demonstration profile `demo_analyst` is permanently protected against deletion (returns HTTP 403 Forbidden).
- Path traversal vectors (`../`, absolute paths, null bytes) are blocked by ID sanitization and validation.

---

## 7. Scientific Limitations & Responsible Use

1. **Classifier Resemblance $\ne$ Proof of Deception:**
   Engine A measures linguistic resemblance to training texts in benchmark research corpora (Ott et al. hotel reviews, Pérez-Rosas & Mihalcea personal statements). A "Deceptive" classification indicates that the text exhibits lexical and n-gram patterns statistically similar to deceptive texts in that specific dataset. It **never proves that an individual lied or that the content is factually false**.

2. **Stylometric Variance $\ne$ Untruthfulness:**
   Engine B quantifies stylistic divergence from an author's historical writing samples across 17 linguistic features. Stylistic variation is common and benign: changes in communication device (mobile vs. desktop), formality, audience, urgency, collaborative editing, or subject matter routinely cause substantial shifts in sentence length, punctuation, and vocabulary. Stylistic divergence **must never be interpreted as evidence of deception**.

3. **Strict Separation of Analysis Engines (No Score Blending):**
   General benchmark classifier scores and personalized stylometric alignment indices measure fundamentally different linguistic attributes. DSP_Prok strictly enforces an **independent metric policy**: they are presented side-by-side with contextual narrative guidance, but are **never mathematically combined into an overall "deception probability" or "composite lie index."**
