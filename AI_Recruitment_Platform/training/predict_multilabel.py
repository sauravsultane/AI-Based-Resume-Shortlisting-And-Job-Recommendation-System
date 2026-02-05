import os
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import json

# Setup paths (Assuming this script is run from project root or training dir)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

def text_cleaning(text):
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def predict_categories_multi(resume_text, threshold=0.3):
    """
    Predict multiple categories for a resume with confidence scores.
    
    Args:
        resume_text (str): The resume text
        threshold (float): Minimum confidence to include category (default 0.3)
    
    Returns:
        dict: {
            'primary_category': str,
            'all_categories': list of {category, confidence},
            'tags': list of category names above threshold
        }
    """
    try:
        # Load multi-label model
        model_path = os.path.join(TRAINING_DIR, 'model_multilabel.pkl')
        vectorizer_path = os.path.join(TRAINING_DIR, 'vectorizer_multilabel.pkl')
        encoder_path = os.path.join(TRAINING_DIR, 'encoder_multilabel.pkl')
        
        if not all([os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]]):
            print("Multi-label model not found. Please train with --multi-label flag first.")
            return None
        
        clf = joblib.load(model_path)
        tfidf = joblib.load(vectorizer_path)
        le = joblib.load(encoder_path)
        
        # Preprocess and vectorize
        cleaned_text = text_cleaning(resume_text)
        X = tfidf.transform([cleaned_text])
        
        # Get probabilities for ALL categories
        probabilities = clf.predict_proba(X)[0]
        
        # Create category-confidence pairs
        all_categories = [
            {'category': cat, 'confidence': float(prob)}
            for cat, prob in zip(le.classes_, probabilities)
        ]
        
        # Sort by confidence (highest first)
        all_categories_sorted = sorted(all_categories, key=lambda x: x['confidence'], reverse=True)
        
        # Primary category (highest confidence)
        primary_category = all_categories_sorted[0]['category']
        
        # Tags (categories above threshold)
        tags = [cat['category'] for cat in all_categories_sorted if cat['confidence'] >= threshold]
        
        return {
            'primary_category': primary_category,
            'all_categories': all_categories_sorted,
            'tags': tags
        }
    
    except Exception as e:
        print(f"Error in multi-label prediction: {e}")
        return None

if __name__ == "__main__":
    # Test multi-label prediction
    sample_resume = """
    Senior Full Stack Developer with 5+ years of experience in Python, Django, React, and Node.js.
    Built scalable web applications serving millions of users. Proficient in both frontend (React, Vue.js)
    and backend (Python, FastAPI, PostgreSQL) development. Experience with ML model deployment using Flask.
    """
    
    result = predict_categories_multi(sample_resume, threshold=0.3)
    if result:
        print("="*60)
        print("MULTI-LABEL PREDICTION TEST")
        print("="*60)
        print(f"\nPrimary Category: {result['primary_category']}")
        print(f"\nAll Category Tags: {', '.join(result['tags'])}")
        print(f"\nDetailed Confidence Scores:")
        for cat in result['all_categories'][:5]:  # Top 5
            print(f"  {cat['category']}: {cat['confidence']*100:.2f}%")
