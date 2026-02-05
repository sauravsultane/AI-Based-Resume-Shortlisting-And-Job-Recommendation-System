from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Applicant(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    resume_file = models.FileField(upload_to='resumes/')
    
    # Personal Details
    phone = models.CharField(max_length=20, blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], blank=True, null=True)
    languages = models.CharField(max_length=255, blank=True, null=True, help_text="Comma-separated languages")
    
    # Education
    education_degree = models.CharField(max_length=100, blank=True, null=True, help_text="e.g. B.Tech Computer Science")
    education_institution = models.CharField(max_length=100, blank=True, null=True, help_text="e.g. SPPU")
    
    # AI Extracted Data
    predicted_category = models.CharField(max_length=100, blank=True, null=True)
    category_scores = models.JSONField(
        blank=True, 
        null=True,
        help_text="Multi-label category predictions with confidence scores: {category: confidence}"
    )
    resume_score = models.IntegerField(default=0)
    page_count = models.IntegerField(default=1)
    experience_level = models.CharField(max_length=50, blank=True, null=True)
    
    # Skills stored as Text (JSON or Comma-separated)
    actual_skills = models.TextField(blank=True, default='[]')
    recommended_skills = models.TextField(blank=True, default='[]')

    
    upload_date = models.DateTimeField(default=timezone.now)

    def calculate_profile_completion(self):
        score = 0
        # Weights
        if self.education_degree or self.education_institution: score += 20
        if self.experience_level: score += 20
        if self.projects.exists(): score += 20
        if self.languages: score += 10
        if self.location: score += 10
        if self.gender: score += 10
        if self.actual_skills and self.actual_skills != '[]': score += 10
        return score

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.predicted_category}"

class Project(models.Model):
    applicant = models.ForeignKey(Applicant, on_delete=models.CASCADE, related_name='projects')
    title = models.CharField(max_length=200)
    link = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.title

class Certification(models.Model):
    applicant = models.ForeignKey(Applicant, on_delete=models.CASCADE, related_name='certifications')
    title = models.CharField(max_length=200)
    link = models.URLField(blank=True, null=True)
    issue_date = models.DateField(blank=True, null=True)
    
    def __str__(self):
        return self.title
