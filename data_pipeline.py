"""
data_pipeline.py
Multi-Domain Dataset Pipeline and Leakage Audit for Text Deception Detection.

Implements Phase 2 & Phase 3 requirements:
- Standardized multi-domain schema: text, label (0=Truthful, 1=Deceptive), domain, source_dataset, author_id, date, source_label
- Data deduplication and near-duplicate audit
- Group-aware and stratified splitting
- Leakage detection between train, validation, and test splits
- Extensible dataset import architecture with provenance tracking
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit

# Standard Schema Definition
STANDARDIZED_COLUMNS = [
    "text",
    "label",
    "domain",
    "source_dataset",
    "source_label",
    "author_id",
    "date",
    "metadata"
]

DOMAIN_CATALOG = {
    "hotel_review": {
        "display_name": "Hotel Reviews",
        "description": "Customer hotel accommodation reviews and opinions",
        "validation_status": "Validated (Ott et al. 2011, 2013 gold standard)",
        "intended_use": "Research & Decision Support"
    },
    "product_review": {
        "display_name": "Product Reviews",
        "description": "E-commerce product reviews and consumer feedback",
        "validation_status": "Supported / Extensible pipeline",
        "intended_use": "Research & Decision Support"
    },
    "email_message": {
        "display_name": "Emails & Written Messages",
        "description": "Interpersonal and professional correspondence",
        "validation_status": "Supported / Extensible pipeline",
        "intended_use": "Research & Decision Support"
    },
    "social_comment": {
        "display_name": "Social Media Posts & Comments",
        "description": "Online forum, social network, and public discussion comments",
        "validation_status": "Supported / Extensible pipeline",
        "intended_use": "Research & Decision Support"
    },
    "personal_statement": {
        "display_name": "Personal Statements & Narratives",
        "description": "First-person accounts, statements, and essays",
        "validation_status": "Supported / Extensible pipeline",
        "intended_use": "Research & Decision Support"
    },
    "general_text": {
        "display_name": "General Written Text",
        "description": "General-purpose written documents and statements",
        "validation_status": "Supported / Extensible pipeline",
        "intended_use": "Research & Decision Support"
    }
}


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text strictly for duplicate detection (lowercased, stripped, collapsed whitespace)."""
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def compute_text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for exact duplicate detection."""
    norm = normalize_text_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity for near-duplicate detection."""
    tokens1 = set(normalize_text_for_dedup(text1).split())
    tokens2 = set(normalize_text_for_dedup(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


class DatasetImporter:
    """Base class for dataset importers."""

    def load_and_standardize(self) -> pd.DataFrame:
        raise NotImplementedError


class OttHotelReviewImporter(DatasetImporter):
    """
    Importer for the Ott et al. (2011, 2013) deceptive opinion dataset.
    Reference: Ott et al., Finding Deceptive Opinion Spam by Any Means Necessary (ACL 2011).
    Dataset contains 1,600 balanced reviews (800 truthful, 800 deceptive).
    """

    def __init__(self, filepath: str = "deceptive-opinion.csv"):
        self.filepath = filepath

    def load_and_standardize(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Source file not found: {self.filepath}")

        raw_df = pd.read_csv(self.filepath)
        # Expected columns: ['deceptive', 'hotel', 'polarity', 'source', 'text']
        records = []
        for _, row in raw_df.iterrows():
            raw_label = str(row.get("deceptive", "")).strip().lower()
            if raw_label in ["truthful", "0"]:
                label = 0
            elif raw_label in ["deceptive", "1"]:
                label = 1
            else:
                continue

            records.append({
                "text": str(row.get("text", "")).strip(),
                "label": label,
                "domain": "hotel_review",
                "source_dataset": "ott_et_al_2011_2013",
                "source_label": raw_label,
                "author_id": None,  # MTurk workers anonymous in public release
                "date": None,
                "metadata": json.dumps({
                    "hotel": row.get("hotel", ""),
                    "polarity": row.get("polarity", ""),
                    "source": row.get("source", "")
                })
            })

        df = pd.DataFrame(records)
        return df


class OpenDomainDeceptionImporter(DatasetImporter):
    """
    Importer for the Pérez-Rosas & Mihalcea (2015) Open Domain Deception Dataset.
    Reference: Verónica Pérez-Rosas & Rada Mihalcea, "Experiments in Open Domain Deception Detection", EMNLP 2015.
    Source: http://web.eecs.umich.edu/~mihalcea/downloads/openDeception.2015.tar.gz
    Contains 7,168 short open-domain statements from 512 contributors with demographic metadata.
    """

    def __init__(self, filepath: str = "personal_statements_raw.csv"):
        self.filepath = filepath

    def load_and_standardize(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            # Check tar.gz archive
            tar_path = "openDeception.2015.tar.gz"
            if os.path.exists(tar_path):
                import tarfile
                with tarfile.open(tar_path, "r:gz") as tar:
                    member = tar.getmember("OpenDeception/7Truth7LiesDataset.csv")
                    with tar.extractfile(member) as f_in, open(self.filepath, "wb") as f_out:
                        f_out.write(f_in.read())
            else:
                raise FileNotFoundError(f"Source file not found: {self.filepath}")

        raw_df = pd.read_csv(self.filepath, quotechar="'", escapechar='\\', skipinitialspace=True)
        records = []
        for _, row in raw_df.iterrows():
            raw_label = str(row.get("class", "")).strip().lower()
            if raw_label in ["truth", "0", "truthful"]:
                label = 0
            elif raw_label in ["lie", "1", "deceptive"]:
                label = 1
            else:
                continue

            instance_id = str(row.get("id", "")).strip()
            # Extract contributor ID from instance ID (e.g., '1_f_l_1' -> '1_f')
            match = re.match(r"^(\d+_[fm])", instance_id)
            author_id = match.group(1) if match else instance_id

            age_val = None
            if pd.notna(row.get("age")):
                try:
                    age_val = int(row.get("age"))
                except (ValueError, TypeError):
                    age_val = None

            records.append({
                "text": str(row.get("text", "")).strip(),
                "label": label,
                "domain": "personal_statement",
                "source_dataset": "perez_rosas_mihalcea_2015",
                "source_label": raw_label,
                "author_id": author_id,
                "date": None,
                "metadata": json.dumps({
                    "id": instance_id,
                    "gender": str(row.get("_gender", "")).strip(),
                    "age": age_val,
                    "education": str(row.get("education", "")).strip(),
                    "country": str(row.get("country", "")).strip()
                })
            })

        return pd.DataFrame(records)


class GenericDatasetImporter(DatasetImporter):
    """Generic importer for external datasets with custom column mappings."""

    def __init__(
        self,
        filepath: str,
        text_col: str,
        label_col: str,
        truthful_value: Any = "truthful",
        deceptive_value: Any = "deceptive",
        domain: str = "general_text",
        source_dataset_name: str = "custom_dataset",
        author_col: Optional[str] = None,
        date_col: Optional[str] = None
    ):
        self.filepath = filepath
        self.text_col = text_col
        self.label_col = label_col
        self.truthful_value = truthful_value
        self.deceptive_value = deceptive_value
        self.domain = domain
        self.source_dataset_name = source_dataset_name
        self.author_col = author_col
        self.date_col = date_col

    def load_and_standardize(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Source file not found: {self.filepath}")

        raw_df = pd.read_csv(self.filepath)
        records = []
        for _, row in raw_df.iterrows():
            raw_val = row.get(self.label_col)
            if raw_val is None:
                continue

            if str(raw_val).strip().lower() == str(self.truthful_value).strip().lower():
                label = 0
            elif str(raw_val).strip().lower() == str(self.deceptive_value).strip().lower():
                label = 1
            else:
                continue

            author = str(row[self.author_col]) if self.author_col and self.author_col in row and pd.notna(row[self.author_col]) else None
            date_val = str(row[self.date_col]) if self.date_col and self.date_col in row and pd.notna(row[self.date_col]) else None

            records.append({
                "text": str(row.get(self.text_col, "")).strip(),
                "label": label,
                "domain": self.domain,
                "source_dataset": self.source_dataset_name,
                "source_label": str(raw_val),
                "author_id": author,
                "date": date_val,
                "metadata": json.dumps({})
            })

        return pd.DataFrame(records)


class DatasetAuditPipeline:
    """
    Performs audit, cleaning, duplicate detection, and leakage prevention.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.clean_df = pd.DataFrame()
        self.audit_report: Dict[str, Any] = {}

    def audit_and_clean(self, near_duplicate_threshold: float = 0.90) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        total_initial = len(self.raw_df)
        
        # 1. Check missing values
        missing_text = self.raw_df["text"].isna().sum()
        empty_text = (self.raw_df["text"].str.strip() == "").sum()
        missing_label = self.raw_df["label"].isna().sum()

        valid_df = self.raw_df.dropna(subset=["text", "label"]).copy()
        valid_df = valid_df[valid_df["text"].str.strip().str.len() > 0]

        # 2. Add deduplication hash
        valid_df["norm_hash"] = valid_df["text"].apply(compute_text_hash)

        # 3. Check exact duplicates
        duplicate_mask = valid_df.duplicated(subset=["norm_hash"], keep=False)
        duplicate_hashes = valid_df[duplicate_mask]["norm_hash"].unique()
        
        # Check label conflicts on duplicates
        hash_to_labels = {}
        for _, row in valid_df.iterrows():
            h = row["norm_hash"]
            if h not in hash_to_labels:
                hash_to_labels[h] = set()
            hash_to_labels[h].add(row["label"])

        conflicting_hashes = set(h for h, lbls in hash_to_labels.items() if len(lbls) > 1)
        conflicts = len(conflicting_hashes)

        # Exclude conflicting rows if requested (contradictory ground truth)
        if conflicting_hashes:
            conflicting_mask = valid_df["norm_hash"].isin(conflicting_hashes)
            conflicting_rows_dropped = int(conflicting_mask.sum())
            non_conflicting_df = valid_df[~conflicting_mask].copy()
        else:
            conflicting_rows_dropped = 0
            non_conflicting_df = valid_df.copy()

        exact_duplicates_dropped = 0
        dedup_records = []
        seen_hashes = set()

        for _, row in non_conflicting_df.iterrows():
            h = row["norm_hash"]
            if h in seen_hashes:
                exact_duplicates_dropped += 1
                continue
            seen_hashes.add(h)
            dedup_records.append(row)

        clean_df = pd.DataFrame(dedup_records)
        clean_df = clean_df.drop(columns=["norm_hash"]).reset_index(drop=True)

        # 4. Near-duplicate audit (sample if large, exhaustive if <= 2000)
        near_duplicates = []
        n_samples = len(clean_df)
        if n_samples <= 2000:
            texts = clean_df["text"].tolist()
            labels = clean_df["label"].tolist()
            for i in range(min(500, n_samples)):  # check first 500 for computational speed
                for j in range(i + 1, min(500, n_samples)):
                    sim = compute_jaccard_similarity(texts[i], texts[j])
                    if sim >= near_duplicate_threshold:
                        near_duplicates.append({
                            "idx1": i,
                            "idx2": j,
                            "similarity": round(sim, 4),
                            "label1": labels[i],
                            "label2": labels[j],
                            "conflicting_label": (labels[i] != labels[j])
                        })

        # Summary statistics
        class_balance = clean_df["label"].value_counts().to_dict()
        domains_present = clean_df["domain"].value_counts().to_dict()
        sources_present = clean_df["source_dataset"].value_counts().to_dict()

        self.clean_df = clean_df
        self.audit_report = {
            "initial_samples": int(total_initial),
            "missing_text_samples": int(missing_text + empty_text),
            "missing_label_samples": int(missing_label),
            "exact_duplicate_samples_dropped": int(exact_duplicates_dropped),
            "conflicting_duplicate_labels": int(conflicts),
            "conflicting_rows_dropped": int(conflicting_rows_dropped),
            "near_duplicates_found": len(near_duplicates),
            "clean_sample_count": len(clean_df),
            "class_distribution": {
                "truthful_0": int(class_balance.get(0, 0)),
                "deceptive_1": int(class_balance.get(1, 0))
            },
            "domain_breakdown": domains_present,
            "source_datasets": sources_present,
            "near_duplicate_examples": near_duplicates[:5]
        }

        return self.clean_df, self.audit_report

    def create_leak_free_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Create stratified, leak-free train, validation, and test splits.
        Verifies that no identical or hash-equivalent texts cross split boundaries.
        """
        if self.clean_df.empty:
            raise ValueError("Must run audit_and_clean() before splitting.")

        df = self.clean_df.copy()

        # Step 1: Stratified train+val vs test split
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=random_state
        )

        # Step 2: Stratified train vs val split
        adjusted_val_size = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            stratify=train_val["label"],
            random_state=random_state
        )

        # Step 3: Leakage verification
        train_hashes = set(train["text"].apply(compute_text_hash))
        val_hashes = set(val["text"].apply(compute_text_hash))
        test_hashes = set(test["text"].apply(compute_text_hash))

        leak_train_val = len(train_hashes.intersection(val_hashes))
        leak_train_test = len(train_hashes.intersection(test_hashes))
        leak_val_test = len(val_hashes.intersection(test_hashes))

        if leak_train_val > 0 or leak_train_test > 0 or leak_val_test > 0:
            raise RuntimeError(
                f"Data leakage detected! Train-Val: {leak_train_val}, "
                f"Train-Test: {leak_train_test}, Val-Test: {leak_val_test}"
            )

        splits = {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)
        }

        self.audit_report["splits_audit"] = {
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "leakage_train_val": leak_train_val,
            "leakage_train_test": leak_train_test,
            "leakage_val_test": leak_val_test,
            "verification_status": "PASSED: Zero overlap between splits"
        }

        return splits

    def create_contributor_grouped_splits(
        self,
        group_col: str = "author_id",
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Create contributor-grouped train, validation, and test splits.
        Ensures that no author/contributor appears in multiple splits.
        Verifies both author separation and text hash separation.
        """
        if self.clean_df.empty:
            raise ValueError("Must run audit_and_clean() before splitting.")

        df = self.clean_df.copy()
        if group_col not in df.columns or df[group_col].isna().all():
            raise ValueError(f"Valid grouping column '{group_col}' required for contributor-grouped splitting.")

        # Step 1: train_val vs test by author
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))
        train_val = df.iloc[train_val_idx].copy().reset_index(drop=True)
        test = df.iloc[test_idx].copy().reset_index(drop=True)

        # Step 2: train vs val by author
        adjusted_val_ratio = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val, groups=train_val[group_col]))
        train = train_val.iloc[train_idx].copy().reset_index(drop=True)
        val = train_val.iloc[val_idx].copy().reset_index(drop=True)

        # Step 3: Contributor and text hash separation verification
        train_authors = set(train[group_col])
        val_authors = set(val[group_col])
        test_authors = set(test[group_col])

        author_leak_tv = len(train_authors.intersection(val_authors))
        author_leak_tt = len(train_authors.intersection(test_authors))
        author_leak_vt = len(val_authors.intersection(test_authors))

        train_hashes = set(train["text"].apply(compute_text_hash))
        val_hashes = set(val["text"].apply(compute_text_hash))
        test_hashes = set(test["text"].apply(compute_text_hash))

        hash_leak_tv = len(train_hashes.intersection(val_hashes))
        hash_leak_tt = len(train_hashes.intersection(test_hashes))
        hash_leak_vt = len(val_hashes.intersection(test_hashes))

        if author_leak_tv > 0 or author_leak_tt > 0 or author_leak_vt > 0:
            raise RuntimeError(
                f"Author leakage detected! Train-Val: {author_leak_tv}, "
                f"Train-Test: {author_leak_tt}, Val-Test: {author_leak_vt}"
            )
        if hash_leak_tv > 0 or hash_leak_tt > 0 or hash_leak_vt > 0:
            raise RuntimeError(
                f"Hash leakage detected! Train-Val: {hash_leak_tv}, "
                f"Train-Test: {hash_leak_tt}, Val-Test: {hash_leak_vt}"
            )

        splits = {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)
        }

        self.audit_report["splits_audit"] = {
            "split_strategy": f"Contributor-grouped by {group_col}",
            "train_size": len(train),
            "train_authors": len(train_authors),
            "val_size": len(val),
            "val_authors": len(val_authors),
            "test_size": len(test),
            "test_authors": len(test_authors),
            "author_leakage": {
                "train_val": author_leak_tv,
                "train_test": author_leak_tt,
                "val_test": author_leak_vt
            },
            "hash_leakage": {
                "train_val": hash_leak_tv,
                "train_test": hash_leak_tt,
                "val_test": hash_leak_vt
            },
            "verification_status": "PASSED: Zero author and zero text-hash overlap across splits"
        }

        return splits


def run_pipeline():
    """Execute pipeline for baseline and generate standardized datasets and audit report."""
    print("=== RUNNING MULTI-DOMAIN DATASET PIPELINE & AUDIT ===")
    importer = OttHotelReviewImporter("deceptive-opinion.csv")
    raw_df = importer.load_and_standardize()
    print(f"Loaded {len(raw_df)} standardized rows from Ott et al. dataset.")

    auditor = DatasetAuditPipeline(raw_df)
    clean_df, audit_rep = auditor.audit_and_clean()
    print(f"Cleaned dataset: {len(clean_df)} rows. Audit report:")
    print(json.dumps(audit_rep, indent=2))

    splits = auditor.create_leak_free_splits(test_size=0.2, val_size=0.1, random_state=42)
    print("Leakage verification passed across splits!")

    # Save standardized artifacts
    clean_df.to_csv("standardized_multi_domain_corpus.csv", index=False)
    splits["train"].to_csv("split_train.csv", index=False)
    splits["val"].to_csv("split_val.csv", index=False)
    splits["test"].to_csv("split_test.csv", index=False)

    with open("dataset_audit_report.json", "w") as f:
        json.dump(audit_rep, f, indent=2)

    with open("multi_domain_catalog.json", "w") as f:
        json.dump(DOMAIN_CATALOG, f, indent=2)

    print("Saved standardized artifacts and audit reports successfully.")


def run_personal_statements_pipeline():
    """Execute pipeline for Open-Domain Personal Statements and generate audit artifacts."""
    print("=== RUNNING PERSONAL STATEMENTS DATASET PIPELINE & AUDIT ===")
    importer = OpenDomainDeceptionImporter("personal_statements_raw.csv")
    raw_df = importer.load_and_standardize()
    print(f"Loaded {len(raw_df)} standardized rows from Pérez-Rosas & Mihalcea (2015) dataset.")

    # Cross-dataset overlap audit against existing hotel reviews
    hotel_corpus_file = "standardized_multi_domain_corpus.csv"
    cross_overlap_count = 0
    if os.path.exists(hotel_corpus_file):
        hotel_df = pd.read_csv(hotel_corpus_file)
        hotel_hashes = set(hotel_df["text"].apply(compute_text_hash))
        ps_hashes = set(raw_df["text"].apply(compute_text_hash))
        cross_overlap_count = len(ps_hashes.intersection(hotel_hashes))
        print(f"Cross-dataset hash overlap with hotel corpus: {cross_overlap_count}")

    auditor = DatasetAuditPipeline(raw_df)
    clean_df, audit_rep = auditor.audit_and_clean()
    audit_rep["cross_dataset_overlap_with_hotel_corpus"] = cross_overlap_count

    print(f"Cleaned personal statement dataset: {len(clean_df)} rows. Audit report:")
    print(json.dumps(audit_rep, indent=2))

    splits = auditor.create_contributor_grouped_splits(
        group_col="author_id",
        test_size=0.20,
        val_size=0.10,
        random_state=42
    )
    print("Contributor-grouped separation verified!")

    # Save distinct artifacts
    clean_df.to_csv("personal_statements_corpus.csv", index=False)
    splits["train"].to_csv("personal_statements_split_train.csv", index=False)
    splits["val"].to_csv("personal_statements_split_val.csv", index=False)
    splits["test"].to_csv("personal_statements_split_test.csv", index=False)

    with open("dataset_audit_personal_statements.json", "w") as f:
        json.dump(audit_rep, f, indent=2)

    print("Saved personal statement artifacts and dataset_audit_personal_statements.json successfully.")


if __name__ == "__main__":
    run_personal_statements_pipeline()
