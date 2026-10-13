"""
Optimized Experiment & Model Selection Benchmark
Performs 5-Fold Stratified Cross-Validation on the training dataset.
Ensures zero data leakage: Vectorizers and augmentations are fit/generated strictly inside each fold.
Includes lru_cache for high performance and flush=True for real-time reporting.
"""

import pandas as pd
import numpy as np
import re
import random
from functools import lru_cache
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import wordnet

# Load original train dataset
train_df = pd.read_csv('train_original.csv')
print(f"Loaded train_df: {train_df.shape}", flush=True)

# Also check leakage removal: remove the 1 review that matches test index 167
leak_free_train = train_df[train_df['text_clean'] != "my daughter and i woke in the morning wanting to go swimming. when we arrived at the pool the water was covered by a white scum. you could not see through the water. we went back to our room disappointed and wanting to shower. our room was not any better. it smelled like old smoke. the paint in the bathroom was peeling and there was what looked like mold on the wall. we called room service to order dinner and breakfast. both meals took over 2 hours to arrive. to top off our stay, the fire alarm went off twice in the night for no reason. upon checkout we were given an extra day's room charge that took 45 minutes to correct. overall this was the worst hotel experience i have ever had."].copy().reset_index(drop=True)
print(f"Leak-free train_df shape: {leak_free_train.shape}", flush=True)

PARAPHRASE_RULES = [
    (r"\bvery good\b", "really good"),
    (r"\bvery nice\b", "really nice"),
    (r"\bvery clean\b", "extremely clean"),
    (r"\bvery comfortable\b", "quite comfortable"),
    (r"\bvery helpful\b", "extremely helpful"),
    (r"\bvery friendly\b", "really friendly"),
    (r"\bvery beautiful\b", "really beautiful"),
    (r"\bvery pleasant\b", "really pleasant"),
    (r"\bvery bad\b", "really poor"),
    (r"\bvery poor\b", "quite poor"),
    (r"\breally good\b", "very good"),
    (r"\breally nice\b", "very nice"),
    (r"\breally clean\b", "very clean"),
    (r"\breally comfortable\b", "very comfortable"),
    (r"\breally friendly\b", "very friendly"),
    (r"\breally helpful\b", "very helpful"),
    (r"\bquite good\b", "fairly good"),
    (r"\bquite nice\b", "fairly nice"),
    (r"\bquite clean\b", "fairly clean"),
    (r"\ba lot\b", "quite a lot"),
    (r"\bI loved\b", "I really liked"),
    (r"\bI love\b", "I really like"),
    (r"\bI liked\b", "I enjoyed"),
    (r"\bI did not like\b", "I disliked"),
    (r"\bI didn't like\b", "I disliked"),
    (r"\bwould definitely\b", "would certainly"),
    (r"\bdefinitely recommend\b", "certainly recommend"),
    (r"\bhighly recommend\b", "strongly recommend"),
    (r"\bexcellent service\b", "great service"),
    (r"\bgreat service\b", "excellent service"),
    (r"\bgreat experience\b", "excellent experience"),
    (r"\bexcellent experience\b", "great experience"),
    (r"\bgreat hotel\b", "excellent hotel"),
    (r"\bexcellent hotel\b", "great hotel"),
    (r"\bbeautiful room\b", "lovely room"),
    (r"\blovely room\b", "beautiful room"),
    (r"\bgood location\b", "convenient location"),
    (r"\bconvenient location\b", "good location"),
]

def controlled_paraphrase(text):
    new_text = text
    rules = PARAPHRASE_RULES.copy()
    random.shuffle(rules)
    changed = False
    for pattern, replacement in rules:
        if re.search(pattern, new_text, flags=re.IGNORECASE):
            new_text = re.sub(pattern, replacement, new_text, count=1, flags=re.IGNORECASE)
            changed = True
            break
    return new_text if changed else None

@lru_cache(maxsize=10000)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            replacement = lemma.name().replace('_', ' ')
            if replacement.lower() != word.lower() and replacement.isalpha() and 3 <= len(replacement) <= 15:
                synonyms.add(replacement.lower())
    return tuple(synonyms)

def synonym_replace(text):
    words = text.split()
    candidates = []
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if len(clean_word) < 4:
            continue
        synonyms = get_synonyms(clean_word)
        if synonyms:
            candidates.append((i, synonyms))
    if not candidates:
        return text
    index, synonyms = random.choice(candidates)
    replacement = random.choice(synonyms)
    original_word = words[index]
    if original_word[0].isupper():
        replacement = replacement.capitalize()
    words[index] = replacement
    return ' '.join(words)

# Pre-generate augmentation pools for efficiency and determinism
random.seed(42)
pre_synonyms = [synonym_replace(t) for t in leak_free_train['text_clean']]
pre_paraphrases = [controlled_paraphrase(t) for t in leak_free_train['text_clean']]

def run_cv(train_data, feature_type='combined', aug_type='none', C=0.15, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, precs, recs, f1s = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data['text_clean'], train_data['label'])):
        fold_train = train_data.iloc[train_idx].copy()
        fold_val = train_data.iloc[val_idx].copy()
        
        # In-fold augmentation strictly from train_idx only
        if aug_type == 'synonym':
            syn_rows = [{'text_clean': pre_synonyms[i], 'label': train_data.iloc[i]['label']} for i in train_idx if pre_synonyms[i] != train_data.iloc[i]['text_clean']]
            fold_train = pd.concat([fold_train, pd.DataFrame(syn_rows)], ignore_index=True).drop_duplicates(subset=['text_clean'])
        elif aug_type == 'paraphrase':
            para_rows = [{'text_clean': pre_paraphrases[i], 'label': train_data.iloc[i]['label']} for i in train_idx if pre_paraphrases[i] is not None]
            fold_train = pd.concat([fold_train, pd.DataFrame(para_rows)], ignore_index=True).drop_duplicates(subset=['text_clean'])
        elif aug_type == 'both':
            syn_rows = [{'text_clean': pre_synonyms[i], 'label': train_data.iloc[i]['label']} for i in train_idx if pre_synonyms[i] != train_data.iloc[i]['text_clean']]
            para_rows = [{'text_clean': pre_paraphrases[i], 'label': train_data.iloc[i]['label']} for i in train_idx if pre_paraphrases[i] is not None]
            fold_train = pd.concat([fold_train, pd.DataFrame(syn_rows + para_rows)], ignore_index=True).drop_duplicates(subset=['text_clean'])
        
        # Fit vectorizers strictly inside fold
        if feature_type in ['word', 'combined']:
            w_vec = TfidfVectorizer(analyzer='word', lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.98, sublinear_tf=True, max_features=40000)
            X_tr_w = w_vec.fit_transform(fold_train['text_clean'])
            X_va_w = w_vec.transform(fold_val['text_clean'])
        if feature_type in ['char', 'combined']:
            c_vec = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=(3,5), min_df=2, max_df=0.98, sublinear_tf=True, max_features=40000)
            X_tr_c = c_vec.fit_transform(fold_train['text_clean'])
            X_va_c = c_vec.transform(fold_val['text_clean'])
            
        if feature_type == 'word':
            X_tr, X_va = X_tr_w, X_va_w
        elif feature_type == 'char':
            X_tr, X_va = X_tr_c, X_va_c
        else:
            X_tr = hstack([X_tr_w, X_tr_c]).tocsr()
            X_va = hstack([X_va_w, X_va_c]).tocsr()
            
        y_tr = fold_train['label'].values
        y_va = fold_val['label'].values
        
        clf = LinearSVC(C=C, random_state=random_state)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        
        accs.append(accuracy_score(y_va, preds))
        precs.append(precision_score(y_va, preds, zero_division=0))
        recs.append(recall_score(y_va, preds, zero_division=0))
        f1s.append(f1_score(y_va, preds, zero_division=0))
        
    return {
        'accuracy': np.mean(accs), 'acc_std': np.std(accs),
        'precision': np.mean(precs), 'precision_std': np.std(precs),
        'recall': np.mean(recs), 'recall_std': np.std(recs),
        'f1': np.mean(f1s), 'f1_std': np.std(f1s)
    }

configs = [
    ('1. Word TF-IDF (No Aug, C=0.15)', 'word', 'none', 0.15),
    ('2. Word TF-IDF (No Aug, C=0.5)', 'word', 'none', 0.5),
    ('3. Word TF-IDF (No Aug, C=1.0)', 'word', 'none', 1.0),
    ('4. Char TF-IDF (No Aug, C=0.15)', 'char', 'none', 0.15),
    ('5. Char TF-IDF (No Aug, C=0.5)', 'char', 'none', 0.5),
    ('6. Char TF-IDF (No Aug, C=1.0)', 'char', 'none', 1.0),
    ('7. Combined Word+Char (No Aug, C=0.10)', 'combined', 'none', 0.10),
    ('8. Combined Word+Char (No Aug, C=0.15)', 'combined', 'none', 0.15),
    ('9. Combined Word+Char (No Aug, C=0.25)', 'combined', 'none', 0.25),
    ('10. Combined Word+Char (No Aug, C=0.50)', 'combined', 'none', 0.50),
    ('11. Combined Word+Char (No Aug, C=1.00)', 'combined', 'none', 1.00),
    ('12. Combined Word+Char (Synonym Aug, C=0.15)', 'combined', 'synonym', 0.15),
    ('13. Combined Word+Char (Paraphrase Aug, C=0.15)', 'combined', 'paraphrase', 0.15),
    ('14. Combined Word+Char (Both Aug, C=0.15)', 'combined', 'both', 0.15),
    ('15. Combined Word+Char (Both Aug, C=0.50)', 'combined', 'both', 0.50),
]

print("\n" + "="*85, flush=True)
print(f"{'Model Architecture':<45} | {'CV Accuracy':<16} | {'CV F1':<16} | {'Precision':<9} | {'Recall':<9}", flush=True)
print("="*85, flush=True)

rows = []
for name, ftype, atype, c in configs:
    res = run_cv(leak_free_train, feature_type=ftype, aug_type=atype, C=c)
    rows.append({'Model': name, **res})
    acc_str = f"{res['accuracy']:.4f} (+/-{res['acc_std']:.3f})"
    f1_str = f"{res['f1']:.4f} (+/-{res['f1_std']:.3f})"
    print(f"{name:<45} | {acc_str:<16} | {f1_str:<16} | {res['precision']:.4f}    | {res['recall']:.4f}", flush=True)

df_results = pd.DataFrame(rows)
df_results.to_csv('experiment_cv_results.csv', index=False)
print("="*85, flush=True)
print("Saved experiment_cv_results.csv", flush=True)
