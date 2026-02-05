"""
Test Multi-Label Classification
Test the newly implemented multi-label prediction feature
"""
import os
import sys
import django

# Setup Django
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Recruitment_Platform.settings')
django.setup()

from app.prediction import ResumeParser

# Test resume with multiple skill sets
test_resume = """
John Doe
Senior Software Engineer

SKILLS:
- Python, Django, Flask, FastAPI
- React, JavaScript, TypeScript, HTML/CSS  
- Machine Learning: TensorFlow, PyTorch, scikit-learn
- Cloud: AWS, Docker, Kubernetes
- Databases: PostgreSQL, MongoDB

EXPERIENCE:
- Built full-stack web applications using Django and React
- Developed ML models for recommendation systems
- Deployed microservices using Docker and Kubernetes on AWS
- Experience with both frontend and backend development

PROJECTS:
- E-commerce platform with React frontend and Django backend
- ML-based image classification system using PyTorch
- RESTful APIs with FastAPI for data analytics
"""

print("="*80)
print("MULTI-LABEL CLASSIFICATION TEST")
print("="*80)

parser = ResumeParser()

# Test multi-label prediction
print("\n📊 Testing Multi-Label Prediction...")
result = parser.predict_categories_multi(test_resume, threshold=0.3)

if result:
    print("\n✅ Multi-Label Prediction Successful!")
    print(f"\n🎯 Primary Category: {result['primary_category']}")
    print(f"\n🏷️  All Matched Categories (above 30% threshold):")
    print(f"   {', '.join(result['tags'])}")
    
    print(f"\n📈 Detailed Confidence Scores (Top 5):")
    for i, cat in enumerate(result['all_categories'][:5], 1):
        bar_length = int(cat['confidence'] * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"   {i}. {cat['category']:30s} {bar} {cat['confidence']*100:.1f}%")
    
    print("\n💾 What will be saved to database:")
    print(f"   predicted_category: '{result['primary_category']}'")
    print(f"   category_scores: {{")
    for cat in result['all_categories'][:3]:
        print(f"      '{cat['category']}': {cat['confidence']:.4f},")
    print(f"      ... ({len(result['all_categories'])} total)")
    print(f"   }}")
else:
    print("\n❌ Multi-Label Prediction Failed!")
    print("   Make sure the multi-label model is trained.")
    print("   Run: python training/train_model.py --multi-label")

print("\n" + "="*80)
