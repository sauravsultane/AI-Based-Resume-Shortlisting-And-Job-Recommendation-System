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

# Setup paths (Assuming this script is run from project root or training dir)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAINING_DIR = os.path.join(BASE_DIR, 'training')

def text_cleaning(text):
    # Simplified version for training
    import re
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def train():
    resume_path = os.path.join(DATASET_DIR, "gpt_dataset.csv")
    print(f"Loading GPT-generated dataset from {resume_path}...")
    
    if not os.path.exists(resume_path):
        print("gpt_dataset.csv not found in dataset folder.")
        return

    try:
        df = pd.read_csv(resume_path)
        print(f"Loaded {len(df)} resume records.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # GPT dataset has 'Resume' column instead of 'Resume_str'
    if 'Resume' not in df.columns or 'Category' not in df.columns:
        print("Error: Expected 'Resume' and 'Category' columns in gpt_dataset.csv")
        return
    
    print("Preprocessing data...")
    df['Cleaned_Resumes'] = df['Resume'].apply(text_cleaning)
    
    X = df['Cleaned_Resumes']
    y = df['Category']
    
    # Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Vectorization (Enhanced with more features and bigrams)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)
    
    # Train-Test Split for Evaluation
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_enc, test_size=0.2, random_state=42)
    
    # Handle Class Imbalance with SMOTE
    print("Applying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"Training samples after SMOTE: {X_train_balanced.shape[0]} (was {X_train.shape[0]})")
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data.")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Hyperparameter Tuning with RandomizedSearchCV
    print("Performing hyperparameter tuning...")
    param_dist = {
        'n_estimators': [200, 300, 400],
        'max_depth': [50, 70, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf = RandomizedSearchCV(
        rf_base, param_dist, n_iter=20, cv=3, 
        random_state=42, n_jobs=-1, verbose=1
    )
    
    print("Training Model with optimized hyperparameters...")
    clf.fit(X_train_balanced, y_train_balanced)
    
    print(f"Best parameters found: {clf.best_params_}")
    print(f"Best cross-validation score: {clf.best_score_*100:.2f}%")
    
    # Evaluation
    print("Evaluating Model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Generate Visualizations
    print("Generating visualizations...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(TRAINING_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {TRAINING_DIR}/confusion_matrix.png")
    plt.close()
    
    # 2. Per-Class Accuracy Bar Chart
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    categories = list(le.classes_)
    f1_scores = [report[cat]['f1-score'] for cat in categories]
    precisions = [report[cat]['precision'] for cat in categories]
    recalls = [report[cat]['recall'] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(categories))
    width = 0.25
    
    ax.bar([i - width for i in x], precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar([i + width for i in x], f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Job Category')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance by Category\nOverall Accuracy: {accuracy*100:.2f}%')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(TRAINING_DIR, 'performance_by_category.png'), dpi=300, bbox_inches='tight')
    print(f"Saved performance chart to {TRAINING_DIR}/performance_by_category.png")
    plt.close()
    
    # Print Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save Artifacts
    print(f"\nSaving model artifacts to {TRAINING_DIR}...")
    # Use best parameters found during hyperparameter tuning
    best_params = clf.best_params_
    clf_full = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    # Apply SMOTE to full dataset for deployment
    print("Applying SMOTE to full dataset for final model...")
    try:
        smote_full = SMOTE(random_state=42, k_neighbors=3)
        X_full_balanced, y_full_balanced = smote_full.fit_resample(X_tfidf, y_enc)
        clf_full.fit(X_full_balanced, y_full_balanced)
        print(f"Final model trained on {X_full_balanced.shape[0]} samples (SMOTE applied)")
    except:
        clf_full.fit(X_tfidf, y_enc)
        print(f"Final model trained on {X_tfidf.shape[0]} samples (no SMOTE)")
    joblib.dump(clf_full, os.path.join(TRAINING_DIR, 'model.pkl'))
    joblib.dump(tfidf, os.path.join(TRAINING_DIR, 'vectorizer.pkl'))
    joblib.dump(le, os.path.join(TRAINING_DIR, 'encoder.pkl'))
    
    print("\nTraining Complete!")
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

def train_multi_label():
    """Train model for multi-label classification with probability outputs."""
    import json
    resume_path = os.path.join(DATASET_DIR, "gpt_dataset.csv")
    print(f"[MULTI-LABEL MODE] Loading {resume_path}...")
    
    if not os.path.exists(resume_path):
        print("gpt_dataset.csv not found.")
        return
    
    df = pd.read_csv(resume_path)
    print(f"Loaded {len(df)} resumes. Preprocessing...")
    
    df['Cleaned_Resumes'] = df['Resume'].apply(text_cleaning)
    X, y = df['Cleaned_Resumes'], df['Category']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_enc, test_size=0.2, random_state=42)
    
    print("Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except:
        pass
    
    print("Training Random Forest for multi-label...")
    clf = RandomForestClassifier(n_estimators=300, max_depth=70, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Test probability output
    sample_probs = clf.predict_proba(tfidf.transform([X.iloc[0]]))[0]
    print("\nSample Probability Distribution:")
    for cat, prob in sorted(zip(le.classes_, sample_probs), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {cat}: {prob*100:.1f}%")
    
    # Final training on full dataset
    print("\nTraining final model on full dataset...")
    try:
        smote_full = SMOTE(random_state=42, k_neighbors=3)
        X_full, y_full = smote_full.fit_resample(X_tfidf, y_enc)
    except:
        X_full, y_full = X_tfidf, y_enc
    
    clf_full = RandomForestClassifier(n_estimators=300, max_depth=70, random_state=42, n_jobs=-1)
    clf_full.fit(X_full, y_full)
    
    # Save with _multilabel suffix
    print(f"\nSaving multi-label model to {TRAINING_DIR}...")
    joblib.dump(clf_full, os.path.join(TRAINING_DIR, 'model_multilabel.pkl'))
    joblib.dump(tfidf, os.path.join(TRAINING_DIR, 'vectorizer_multilabel.pkl'))
    joblib.dump(le, os.path.join(TRAINING_DIR, 'encoder_multilabel.pkl'))
    
    metadata = {
        'model_type': 'multi-label',
        'categories': le.classes_.tolist(),
        'accuracy': float(accuracy),
        'default_threshold': 0.3
    }
    with open(os.path.join(TRAINING_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Multi-Label Training Complete!")
    print(f"Categories: {len(le.classes_)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--multi-label':
        train_multi_label()
    else:
        train()

