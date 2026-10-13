"""
profile_manager.py
Manages Personal Writing Profiles and Historical Samples for Engine B.

Provides:
- CRUD for personal profiles and historical writing samples
- Automatic baseline statistics computation upon sample mutation
- Local file isolation in `profiles_store/`
- Privacy guarantees: deletion permanently purges sample text
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from stylometry_engine import BaselineAggregator

PROFILES_DIR = "profiles_store"

class ProfileManager:
    """Manages profile persistence and historical writing sample baselines."""

    def __init__(self, storage_dir: str = PROFILES_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._ensure_sample_profiles()

    def _get_profile_path(self, profile_id: str) -> Optional[str]:
        # Sanitize profile_id to avoid path traversal
        if not profile_id:
            return None
        safe_id = "".join(c for c in str(profile_id) if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            return None
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def _ensure_sample_profiles(self):
        """Seed a representative sample profile if no profiles exist yet."""
        if not os.listdir(self.storage_dir):
            demo_id = "demo_analyst"
            demo_samples = [
                {
                    "id": "samp-1",
                    "text": (
                        "Good morning team. I have reviewed the comprehensive quarterly metrics and consolidated "
                        "all department data into the executive briefing sheet. Overall revenue expanded by 14.2% "
                        "year-over-year, primarily driven by enterprise renewals and expanding accounts. Please review "
                        "the attached financial table and share any feedback prior to Thursday's board committee meeting."
                    ),
                    "date": "2026-08-10",
                    "context": "professional_email",
                    "source_description": "Quarterly Financial Briefing Email",
                    "is_representative": True,
                    "created_at": "2026-08-10T09:00:00"
                },
                {
                    "id": "samp-2",
                    "text": (
                        "Regarding the customer satisfaction feedback report, our analysis identified three primary "
                        "operational friction areas: user onboarding complexity, peak response latency in chat queues, "
                        "and enterprise invoice transparency. I suggest we schedule a 30-minute cross-functional sync "
                        "with engineering leads on Tuesday to coordinate remediation milestones."
                    ),
                    "date": "2026-08-25",
                    "context": "professional_email",
                    "source_description": "Customer Operations Sync Email",
                    "is_representative": True,
                    "created_at": "2026-08-25T14:30:00"
                },
                {
                    "id": "samp-3",
                    "text": (
                        "Following up on our vendor contract renegotiation, the newly proposed terms appear fully acceptable "
                        "with one minor exception in section four regarding third-party warranty liabilities. I have "
                        "marked up the clauses and returned the redlined draft to outside legal counsel for formal signoff."
                    ),
                    "date": "2026-09-12",
                    "context": "professional_email",
                    "source_description": "Contract Review Follow-up",
                    "is_representative": True,
                    "created_at": "2026-09-12T11:15:00"
                },
                {
                    "id": "samp-4",
                    "text": (
                        "Attached is the finalized technical architecture specification for the enterprise cloud migration. "
                        "Our engineering teams benchmarked multiple cloud platforms against strict latency and redundancy "
                        "service level agreements. Our joint recommendation is to adopt the hybrid multi-region architecture "
                        "detailed extensively in section six."
                    ),
                    "date": "2026-09-28",
                    "context": "professional_email",
                    "source_description": "Architecture Evaluation Brief",
                    "is_representative": True,
                    "created_at": "2026-09-28T16:00:00"
                },
                {
                    "id": "samp-5",
                    "text": (
                        "Thank you for attending yesterday's project sprint retrospective. Our open discussion "
                        "highlighted commendable velocity across the platform infrastructure workstreams. I have "
                        "summarized the core action items below and assigned designated team owners to track delivery."
                    ),
                    "date": "2026-10-05",
                    "context": "professional_email",
                    "source_description": "Post-Retrospective Summary Email",
                    "is_representative": True,
                    "created_at": "2026-10-05T10:00:00"
                }
            ]
            self.create_profile(
                name="Alex Vance (Professional Sample)",
                description="Technical Product Analyst with professional corporate correspondence",
                notes="Representative baseline of formal business correspondence",
                user_authorization_confirmed=True,
                profile_id=demo_id,
                initial_samples=demo_samples
            )

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List summary information for all stored profiles."""
        summaries = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        baseline = data.get("baseline", {})
                        sample_count = len(data.get("samples", []))
                        total_words = baseline.get("total_word_count", 0)
                        is_demo = bool(data.get("is_demo", data.get("id") == "demo_analyst"))
                        is_ready = bool(baseline.get("is_operationally_ready", sample_count >= 3 and total_words >= 200))
                        readiness_status = baseline.get(
                            "readiness_status",
                            "Operationally Ready for Comparison" if is_ready else "Needs More Writing Samples"
                        )
                        summaries.append({
                            "id": data["id"],
                            "name": data["name"],
                            "description": data.get("description", ""),
                            "is_demo": is_demo,
                            "sample_count": sample_count,
                            "total_words": total_words,
                            "reliability_rating": baseline.get("reliability_rating", "Unknown"),
                            "reliability_badge": baseline.get("reliability_badge", "secondary"),
                            "is_operationally_ready": is_ready,
                            "readiness_status": readiness_status,
                            "min_samples_needed": baseline.get("min_samples_needed", max(0, 3 - sample_count)),
                            "min_words_needed": baseline.get("min_words_needed", max(0, 200 - total_words)),
                            "created_at": data.get("created_at", "")
                        })
                except Exception:
                    continue
        summaries.sort(key=lambda x: x["created_at"], reverse=True)
        return summaries

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full profile details including samples and computed baseline."""
        fpath = self._get_profile_path(profile_id)
        if not fpath or not os.path.exists(fpath):
            return None
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["is_demo"] = bool(data.get("is_demo", data.get("id") == "demo_analyst"))
            # Ensure baseline operational readiness fields are present even if cached from older format
            b = data.get("baseline", {})
            sc = len(data.get("samples", []))
            tw = b.get("total_word_count", 0)
            if "is_operationally_ready" not in b:
                b["is_operationally_ready"] = (sc >= 3 and tw >= 200)
                b["readiness_status"] = "Operationally Ready for Comparison" if b["is_operationally_ready"] else "Needs More Writing Samples"
                b["readiness_badge"] = "success" if b["is_operationally_ready"] else "warning"
                b["min_samples_needed"] = max(0, 3 - sc)
                b["min_words_needed"] = max(0, 200 - tw)
                b["scientific_limitation_note"] = (
                    "Operational Readiness vs Scientific Validity Notice: Meeting the minimum operational threshold allows the engine to calculate mathematical deviations (Z-scores). "
                    "However, operational readiness does NOT constitute proof of universal scientific reliability. Human writing style varies across communication channels, audience, urgency, and emotional states."
                )
            return data

    def create_profile(
        self,
        name: str,
        description: str = "",
        notes: str = "",
        user_authorization_confirmed: bool = True,
        profile_id: Optional[str] = None,
        initial_samples: Optional[List[Dict[str, Any]]] = None,
        is_demo: bool = False
    ) -> Dict[str, Any]:
        """Create a new profile with optional initial samples and authorization validation."""
        pid = profile_id or f"prof_{uuid.uuid4().hex[:10]}"
        now = datetime.now().isoformat()
        formatted_samples = []
        for s in (initial_samples or []):
            if isinstance(s, dict) and s.get("text"):
                formatted_samples.append({
                    "id": s.get("id") or f"samp_{uuid.uuid4().hex[:8]}",
                    "text": s["text"].strip(),
                    "date": s.get("date"),
                    "context": s.get("context", "general"),
                    "source_description": s.get("source_description", ""),
                    "is_representative": s.get("is_representative", True),
                    "created_at": s.get("created_at") or now
                })

        baseline = BaselineAggregator.compute_baseline(formatted_samples)

        profile_data = {
            "id": pid,
            "name": name,
            "description": description,
            "notes": notes,
            "user_authorization_confirmed": user_authorization_confirmed,
            "is_demo": is_demo or (pid == "demo_analyst"),
            "created_at": now,
            "updated_at": now,
            "samples": formatted_samples,
            "baseline": baseline
        }

        fpath = self._get_profile_path(pid)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        return profile_data

    def delete_profile(self, profile_id: str) -> bool:
        """Permanently delete profile and all associated historical text."""
        if profile_id == "demo_analyst":
            return False
        fpath = self._get_profile_path(profile_id)
        if fpath and os.path.exists(fpath):
            os.remove(fpath)
            return True
        return False

    def add_sample(
        self,
        profile_id: str,
        text: str,
        date: Optional[str] = None,
        context: str = "general",
        source_description: str = "",
        is_representative: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Add a historical sample and recompute the statistical baseline."""
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        sample_id = f"samp_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()
        new_sample = {
            "id": sample_id,
            "text": text.strip(),
            "date": date or now[:10],
            "context": context,
            "source_description": source_description,
            "is_representative": is_representative,
            "created_at": now
        }

        profile["samples"].append(new_sample)
        profile["updated_at"] = now
        profile["baseline"] = BaselineAggregator.compute_baseline(profile["samples"])

        fpath = self._get_profile_path(profile_id)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        return profile

    def delete_sample(self, profile_id: str, sample_id: str) -> Optional[Dict[str, Any]]:
        """Delete an individual sample and update the baseline."""
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        initial_len = len(profile["samples"])
        profile["samples"] = [s for s in profile["samples"] if s["id"] != sample_id]
        if len(profile["samples"]) == initial_len:
            return None

        profile["updated_at"] = datetime.now().isoformat()
        profile["baseline"] = BaselineAggregator.compute_baseline(profile["samples"])

        fpath = self._get_profile_path(profile_id)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        return profile


if __name__ == "__main__":
    pm = ProfileManager()
    profs = pm.list_profiles()
    print("Available profiles:", [p["name"] for p in profs])
    demo = pm.get_profile("demo_analyst")
    print(f"Demo profile samples: {len(demo['samples'])}, Reliability: {demo['baseline']['reliability_rating']}")
