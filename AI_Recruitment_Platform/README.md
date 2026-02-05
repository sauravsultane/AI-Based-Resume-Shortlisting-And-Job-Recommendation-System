# AI Recruitment Platform

## Overview
A production-ready AI-based recruitment platform designed to automate resume shortlisting, categorization, and job recommendations. This system utilizes Machine Learning (NLP, TF-IDF, KNN) to match candidates with suitable job roles and provides an HR dashboard for efficient hiring.

## 🚀 Features
*   **AI Resume Analysis**: Parses PDF resumes, extracts text, and predicts job categories.
*   **Skill Extraction**: Identifies actual skills and recommends missing skills for the specific role.
*   **Resume Scoring**: Algorithmic scoring based on content sections (Experience, Projects, Education) and length.
*   **Job Recommendation**: Auto-suggests jobs based on candidate profile.
*   **HR Dashboard**: Post jobs, view applicants, and trigger AI scraping.
*   **Job Scraper**: Integrated Selenium scraper to fetch real job listings from Naukri (Headless Mode).

## 🛠 Tech Stack
*   **Backend**: Python 3.x, Django 4.x
*   **Database**: SQLite (Development), Scalable to PostgreSQL
*   **ML & NLP**: Scikit-Learn, Pandas, NumPy, SpaCy, NLTK
*   **Front-end**: HTML5, Bootstrap 5
*   **Tools**: Selenium (Scraping), PyPDF2 (PDF Parsing)

## 📂 Project Structure
```text
AI_Recruitment_Platform/
├── app/                 # Candidate & AI Logic (Views, Models, Prediction)
├── dashboard/           # HR Dashboard & Job Management
├── project_config/      # Django Settings & URL Routing
├── templates/           # HTML Templates (Bootstrap UI)
├── static/              # CSS/JS Assets
├── media/               # User Uploads (Resumes)
├── dataset/             # Training Datasets (CSV)
├── training/            # ML Scripts & Artifacts (.pkl files)
└── manage.py            # Entry point
```

## ⚙️ Setup Instructions

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **Initialize Database**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

3.  **Train AI Model**
    Before running the server, generate the ML artifacts:
    ```bash
    python training/train_model.py
    ```

4.  **Run Server**
    ```bash
    python manage.py runserver
    ```

5.  **Access**
    *   **Home**: `http://127.0.0.1:8000/`
    *   **HR Dashboard**: Create a superuser (`python manage.py createsuperuser`) and login.

## 🧠 System Architecture

### 1. Data Ingestion Layer
*   **Resumes**: Uploaded via `ResumeUploadForm`, stored in `media`.
*   **Job Data**: Entered manually by HR or scraped via `run_scraper` (Selenium).

### 2. Processing Layer (AI Engine)
*   **`ResumeParser` Class**: Singleton in `app/prediction.py`.
    *   **Preprocessing**: Cleaning text, removing PII (Bias mitigation).
    *   **Vectorization**: TF-IDF transforms text to numerical vectors.
    *   **Prediction**: `OneVsRestClassifier (KNN)` predicts Job Category.
    *   **Scoring**: Heuristic ruleset evaluates resume completeness.

### 3. Application Layer (Django)
*   **MVT Pattern**: Models define schema, Views handle logic, Templates render UI.
*   **Security**: CSRF protection, Authentication decorators, Password hashing.

## ⚖️ Ethical AI & Bias Mitigation
This system is designed with fairness in mind:
1.  **PII Removal**: The `clean_text` function strips pronouns (he/she) and gender markers before analysis to prevent gender bias in scoring.
2.  **Skill-Based Matching**: Evaluation is primarily driven by skill keyword matching rather than university names or demographic indicators.
3.  **Explainability**: The "Results" page explicitly lists "Actual Skills" detected, showing the candidate *why* they were categorized or scored a certain way.

## 🔮 Future Improvements
*   Replace SQLite with PostgreSQL for production.
*   Implement Deep Learning (BERT/Transformers) for semantic matching.
*   Add email notification service (Celery + Redis).
