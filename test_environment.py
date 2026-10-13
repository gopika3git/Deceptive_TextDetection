import pandas as pd
import numpy as np
import sklearn
import scipy
import spacy
import textstat
import joblib
import nltk

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("spaCy:", spacy.__version__)
print("SciPy:", scipy.__version__)

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

print("\nSUCCESS!")
print("All required libraries are working.")
print("spaCy English model loaded successfully.")