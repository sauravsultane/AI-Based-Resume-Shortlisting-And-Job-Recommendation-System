from django.db import models
from django.contrib.auth.models import User
from app.models import Applicant

class Job(models.Model):
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    experience = models.CharField(max_length=100)
    salary = models.CharField(max_length=100)
    description = models.TextField()
    required_skills = models.TextField(help_text="Comma-separated skills")
    
    # Metadata
    is_active = models.BooleanField(default=True)
    posted_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.title} at {self.company}"

class JobApplication(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('SHORTLISTED', 'Shortlisted'),
        ('REJECTED', 'Rejected'),
        ('HIRED', 'Hired'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    applicant_profile = models.ForeignKey(Applicant, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    match_score = models.FloatField(default=0.0)  # AI Match Score
    
    applied_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'job')

    def __str__(self):
        return f"{self.user.username} applied for {self.job.title}"
