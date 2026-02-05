import os
import re
import joblib
import pandas as pd
import numpy as np
import spacy
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Load SpaCy model for advanced NLP (ensure 'en_core_web_sm' is installed)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Warning: SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

class ResumeParser:
    def __init__(self):
        self.model_path = settings.BASE_DIR / 'training' / 'model.pkl'
        self.vectorizer_path = settings.BASE_DIR / 'training' / 'vectorizer.pkl'
        self.encoder_path = settings.BASE_DIR / 'training' / 'encoder.pkl'
        
        self.model = None
        self.vectorizer = None
        self.encoder = None
        
        # Initialize BERT Model
        # This will download the model on first run (~80MB)
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        self._load_models()

    def _load_models(self):
        """Safely load trained models if they exist."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.encoder = joblib.load(self.encoder_path)

    def clean_text(self, text):
        """
        Clean resume text: transform to lower, remove URLs, special chars, 
        and bias-sensitive terms (basic heuristics).
        """
        text = text.lower()
        text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
        text = re.sub(r'rt|cc', ' ', text)  # remove RT and cc
        text = re.sub(r'#\S+', '', text)  # remove hashtags
        text = re.sub(r'@\S+', '  ', text)  # remove mentions
        text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
        
        # Basic Bias Mitigation: Remove pronouns
        # In a real system, this would be more robust.
        text = re.sub(r'\b(he|she|him|her|his|hers|mr|mrs|miss|ms)\b', '', text)
        
        return text

    def predict_category(self, text):
        """Predict the job category from the resume text."""
        if not self.model:
            return "Unknown (Model Not Loaded)"
        
        cleaned_text = self.clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)
        category = self.encoder.inverse_transform(prediction)[0]
        return category

    def predict_categories_multi(self, text, threshold=0.3):
        """
        Predict multiple categories for a resume with confidence scores.
        
        Args:
            text (str): Resume text
            threshold (float): Minimum confidence to include category (0.0-1.0)
        
        Returns:
            dict: {
                'primary_category': str (highest confidence),
                'all_categories': list of {'category': str, 'confidence': float},
                'tags': list of category names above threshold
            }
        """
        # Load multi-label model
        multilabel_model_path = settings.BASE_DIR / 'training' / 'model_multilabel.pkl'
        multilabel_vectorizer_path = settings.BASE_DIR / 'training' / 'vectorizer_multilabel.pkl'
        multilabel_encoder_path = settings.BASE_DIR / 'training' / 'encoder_multilabel.pkl'
        
        # Check if multi-label model exists
        if not all([
            os.path.exists(multilabel_model_path),
            os.path.exists(multilabel_vectorizer_path),
            os.path.exists(multilabel_encoder_path)
        ]):
            # Fallback to single-label
            category = self.predict_category(text)
            return {
                'primary_category': category,
                'all_categories': [{'category': category, 'confidence': 1.0}],
                'tags': [category]
            }
        
        # Load multi-label models
        ml_model = joblib.load(multilabel_model_path)
        ml_vectorizer = joblib.load(multilabel_vectorizer_path)
        ml_encoder = joblib.load(multilabel_encoder_path)
        
        # Predict probabilities
        cleaned_text = self.clean_text(text)
        vectorized_text = ml_vectorizer.transform([cleaned_text])
        probabilities = ml_model.predict_proba(vectorized_text)[0]
        
        # Create category-confidence pairs
        all_categories = [
            {'category': cat, 'confidence': float(prob)}
            for cat, prob in zip(ml_encoder.classes_, probabilities)
        ]
        
        # Sort by confidence
        all_categories_sorted = sorted(all_categories, key=lambda x: x['confidence'], reverse=True)
        
        # Primary category
        primary_category = all_categories_sorted[0]['category']
        
        # Tags (above threshold)
        tags = [cat['category'] for cat in all_categories_sorted if cat['confidence'] >= threshold]
        
        return {
            'primary_category': primary_category,
            'all_categories': all_categories_sorted,
            'tags': tags
        }

    def extract_skills(self, text):
        """
        Extract skills using basic dictionary matching + Noun Phrases.
        Note: A real system would use a dedicated Entity Ruler or Skill Ontology.
        """
        # Dictionary of common technical skills
        common_skills = {
            # IT / Web
            'python', 'java', 'django', 'react', 'javascript', 'sql', 'html', 'css', 
            'c++', 'c#', 'node.js', 'typescript', 'angular', 'vue', 'flask', 'fastapi',
            
            # Data Science / AI
            'machine learning', 'data science', 'pandas', 'numpy', 'scikit-learn', 
            'tensorflow', 'pytorch', 'nlp', 'deep learning', 'matplotlib', 'tableau',
            
            # DevOps / Cloud
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 
            'terraform', 'ansible', 'linux', 'bash', 'git',
            
            # Core Engineering
            'autocad', 'solidworks', 'ansys', 'catia', 'matlab', 'simulink', 
            'plc', 'scada', 'embedded systems', 'pcb design', 'iot',
            'hvac', 'thermodynamics', 'fluid mechanics', 'staad pro', 'revit', 
            'structural analysis', 'surveying', 'estimation',
            
            # Soft Skills
            'agile', 'scrum', 'project management', 'communication', 'leadership', 
            'problem solving', 'teamwork'
        }
        
        found_skills = set()
        cleaned_text = self.clean_text(text)
        
        # Simple lookup
        for skill in common_skills:
            if f" {skill} " in f" {cleaned_text} ":
                found_skills.add(skill)
        
        return list(found_skills)

    def calculate_score(self, text, page_count):
        """
        Advanced scoring based on Skills, Education, Impact, and Structure.
        """
        score = 0
        cleaned_text = self.clean_text(text)
        
        # 1. Structure (Max 20)
        sections = {
            'experience': 5, 'work history': 5,
            'education': 5, 'academic': 5,
            'skills': 5, 'competencies': 5,
            'projects': 5
        }
        found_sections = 0
        for section, weight in sections.items():
            if section in cleaned_text:
                found_sections += weight
        score += min(found_sections, 20)

        # 2. Skill Density (Max 40)
        # 2 points per skill found
        skills = self.extract_skills(text)
        skill_points = len(skills) * 3
        score += min(skill_points, 40)
        
        # 3. Education Level (Max 10)
        if any(w in cleaned_text for w in ['phd', 'doctorate']):
            score += 10
        elif any(w in cleaned_text for w in ['master', 'mtech', 'mba', 'msc']):
            score += 8
        elif any(w in cleaned_text for w in ['bachelor', 'btech', 'bsc', 'be ']):
            score += 6
            
        # 4. Impact & Metrics (Max 20)
        # Look for percentages or numbers implying quantfiable results
        metrics = re.findall(r'\d+%', text)
        metric_points = len(metrics) * 4
        score += min(metric_points, 20)
        
        # 5. Length Idealness (Max 10)
        words = len(cleaned_text.split())
        if 300 <= words <= 1200:
            score += 10
        elif words < 300:
            score += 5
        else:
            score += 5
            
        return min(score, 100)

    def generate_suggestions(self, text, score):
        """Generate actionable feedback for the candidate."""
        suggestions = []
        cleaned_text = self.clean_text(text)
        
        # 1. Score-based feedback
        if score < 50:
            suggestions.append("Critically Low Score: Your resume lacks detail. Add more sections.")
        elif score < 70:
            suggestions.append("Improve Structure: Ensure you have clear 'Experience' and 'Projects' sections.")
            
        # 2. Content Checks
        if not re.search(r'\d+%', cleaned_text):
            suggestions.append("Add Metrics: Quantify your achievements (e.g., 'Increased sales by 20%').")
            
        if len(cleaned_text.split()) < 300:
            suggestions.append("Expand Details: Your resume is too short. Elaborate on your roles.")
            
        # 3. Skill Suggestions (Heuristic - recommend based on missing common skills)
        common_missing = []
        for tech in ['git', 'sql', 'communication', 'leadership']:
            if tech not in cleaned_text:
                common_missing.append(tech.title())
        
        if common_missing:
            suggestions.append(f"Consider learning/adding these skills: {', '.join(common_missing)}.")
            
        return suggestions
    
    def get_missing_skills(self, category, text):
        """
        Identify missing critical skills based on the predicted category.
        """
        cleaned_text = self.clean_text(text)
        missing = []
        
        # Skill Map (Category -> Essential Skills)
        skill_map = {
            'Data Science': ['python', 'machine learning', 'sql', 'pandas', 'numpy', 'statistics'],
            'Web Designing': ['html', 'css', 'javascript', 'react', 'responsive design'],
            'HR': ['communication', 'management', 'recruitment', 'leadership', 'excel'],
            'Mechanical': ['autocad', 'solidworks', 'ansys', 'thermodynamics', 'manufacturing'],
            'Java Developer': ['java', 'spring', 'hibernate', 'sql', 'oop'],
            'Python Developer': ['python', 'django', 'flask', 'sql', 'api'],
            'DevOps': ['aws', 'docker', 'kubernetes', 'jenkins', 'linux'],
            'Testing': ['selenium', 'junit', 'manual testing', 'automation', 'sql'],
            'Business Analyst': ['sql', 'excel', 'tableau', 'power bi', 'communication'],
            # New GPT Dataset Categories
            'Backend Developer': ['node.js', 'python', 'django', 'flask', 'rest api', 'sql', 'mongodb'],
            'Cloud Engineer': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'ci/cd'],
            'Data Scientist': ['python', 'machine learning', 'pandas', 'numpy', 'tensorflow', 'sql'],
            'Frontend Developer': ['react', 'javascript', 'html', 'css', 'vue.js', 'typescript'],
            'Full Stack Developer': ['react', 'node.js', 'python', 'django', 'mongodb', 'rest api'],
            'Machine Learning Engineer': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp'],
            'Mobile App Developer (iOS/Android)': ['react native', 'flutter', 'swift', 'kotlin', 'firebase'],
        }
        
        # Normalize category to match keys (basic matching)
        target_skills = []
        if category:
            for key in skill_map:
                if key.lower() in category.lower():
                    target_skills = skill_map[key]
                    break
        
        # Default if no category match
        if not target_skills:
            target_skills = ['communication', 'problem solving', 'teamwork', 'git']
            
        # Find missing
        for skill in target_skills:
            if skill not in cleaned_text:
                missing.append(skill.title())
                
    def extract_contact_info(self, text):
        """
        Extract contact details (Email, Phone) from raw text.
        """
        contact_info = {'email': None, 'phone': None}
        
        # Email Extraction
        # Use raw text because clean_text removes '@'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group(0)
            
        # Phone Extraction
        # Matches: +91 9876543210, 987-654-3210, (123) 456-7890
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group(0).strip()
            
        return contact_info

    def extract_education(self, text):
        """Extract Degree and College using keywords and SpaCy ORG."""
        education = {'degree': None, 'institution': None}
        cleaned_text = self.clean_text(text)
        
        # 1. Degree Extraction (Simple List)
        degrees = ['b.tech', 'b.e.', 'b.sc', 'm.tech', 'm.e.', 'm.sc', 'mba', 'phd', 'bachelor', 'master', 'diploma']
        for deg in degrees:
            if deg in cleaned_text:
                education['degree'] = deg.title()
                break # Take the first found
        
        # 2. Institution Extraction
        # Look for "University" or "College" or try SpaCy
        if nlp:
            doc = nlp(text) # Use raw text for better NER
            for ent in doc.ents:
                if ent.label_ == 'ORG' and ('college' in ent.text.lower() or 'university' in ent.text.lower() or 'institute' in ent.text.lower()):
                    education['institution'] = ent.text
                    break
        
        return education

    def extract_location(self, text):
        """Extract GPE (Geopolitical Entity) as location."""
        if not nlp:
            return None
            
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                return ent.text
        return None

    def _extract_section(self, text, keywords):
        """Helper to extract text block for a specific section based on headers."""
        lines = text.split('\n')
        section_text = []
        is_in_section = False
        
        for line in lines:
            clean_line = line.strip().lower()
            # Start of section?
            if any(k in clean_line for k in keywords) and len(clean_line) < 50: 
                is_in_section = True
                continue
            
            # End of section? (Next likely header)
            if is_in_section:
                common_headers = ['education', 'skills', 'experience', 'work history', 'projects', 'certifications', 'interests', 'reference', 'declaration']
                if any(h in clean_line for h in common_headers) and len(clean_line) < 30 and not any(k in clean_line for k in keywords):
                    break
                section_text.append(line)
        
        return "\n".join(section_text)

    def parse_projects(self, text):
        """Extract project titles/descriptions."""
        keywords = ['projects', 'projects undertaken', 'academic projects']
        section_content = self._extract_section(text, keywords)
        
        if not section_content:
            return []
            
        # Basic parsing: treating non-empty lines as project entries
        # Filtering out short lines to avoid noise
        lines = [line.strip() for line in section_content.split('\n') if len(line.strip()) > 10]
        
        projects = []
        # Take up to 5 lines that look like projects
        for line in lines[:5]:
            projects.append({'title': line[:100], 'description': line})
            
        return projects

    def parse_certifications(self, text):
        """Extract certifications."""
        keywords = ['certifications', 'certificates', 'courses', 'achievements']
        section_content = self._extract_section(text, keywords)
        
        if not section_content:
            return []
            
        lines = [line.strip() for line in section_content.split('\n') if len(line.strip()) > 10]
        
        certs = []
        for line in lines[:5]:
             certs.append({'title': line[:100], 'date': None})
                 
        return certs


    def estimate_experience_level(self, page_count, text, score=0):
        """
        Estimate level based on Years of Exp (Regex), Score, and Keywords.
        """
        cleaned_text = self.clean_text(text)
        
        # 1. Regex for Years of Experience (e.g. "5+ years", "3 years")
        years_found = re.findall(r'(\d+)\+?\s*years?', cleaned_text)
        if years_found:
            # Get max years found (heuristic)
            max_years = max([int(y) for y in years_found if int(y) < 40]) # Cap at 40 to avoid noise
            if max_years >= 5:
                return "Professional"
            elif max_years >= 2:
                return "Intermediate"
            else:
                return "Beginner"
        
        # 2. Fallback to Score & Keywords
        if 'senior' in cleaned_text or 'principal' in cleaned_text or 'manager' in cleaned_text:
            return "Professional"
            
        if score >= 80:
            return "Professional"
        elif score >= 50:
            return "Intermediate"
        else:
            return "Beginner"

    def calculate_match_percentage(self, job_description, job_skills, resume_text):
        """
        Calculate a match percentage (0-100) using BERT Semantic Similarity + Skill Overlap.
        """
        if not resume_text or not job_description:
            return 0.0
            
        # 1. BERT Semantic Match (70% Weight)
        # Encodes meaningful context beyond just keywords
        try:
            embeddings1 = self.bert_model.encode(resume_text, convert_to_tensor=True)
            embeddings2 = self.bert_model.encode(job_description, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
            bert_score = max(0, cosine_score * 100) # Ensure non-negative
        except Exception as e:
            print(f"BERT Error: {e}")
            bert_score = 0

        # 2. Hard Skill Match (30% Weight)
        # Specific technical skills are still crucial
        resume_skills = set(self.extract_skills(resume_text))
        
        if isinstance(job_skills, str):
            target_skills = set(s.strip().lower() for s in job_skills.split(',') if s.strip())
        else:
            target_skills = set(job_skills)
            
        if not target_skills:
            skill_score = 0
        else:
            intersection = resume_skills.intersection(target_skills)
            skill_score = (len(intersection) / len(target_skills)) * 100
            
        # Weighted Final Score
        # We give BERT more weight as it captures experience level, domain, and soft skills context
        final_score = (bert_score * 0.7) + (skill_score * 0.3)
        
        return round(min(final_score, 100), 1)

# Singleton instance
resume_parser = ResumeParser()
