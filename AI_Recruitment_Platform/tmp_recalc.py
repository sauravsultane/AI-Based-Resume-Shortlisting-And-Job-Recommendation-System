import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_config.settings')
django.setup()

from dashboard.models import JobApplication
from app.prediction import resume_parser

apps = JobApplication.objects.all()
for app in apps:
    job = app.job
    applicant = app.applicant_profile
    full_text = applicant.resume_text or applicant.actual_skills
    match_percentage = resume_parser.calculate_match_percentage(
        job_description=job.description,
        job_skills=job.required_skills,
        resume_text=full_text
    )
    app.match_score = match_percentage
    app.save()
    print(f"Updated application {app.id} to new score: {match_percentage}")
