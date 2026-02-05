from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Job, JobApplication
from app.scrape_jobs import run_scraper

@login_required
def hr_dashboard(request):
    # Only allow superusers or specific group
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    jobs = Job.objects.all().order_by('-posted_date')
    
    # Calculate Stats
    total_applicants = JobApplication.objects.count()
    shortlisted_count = JobApplication.objects.filter(status='SHORTLISTED').count()
    
    return render(request, 'hr_dashboard.html', {
        'jobs': jobs,
        'total_applicants': total_applicants,
        'shortlisted_count': shortlisted_count
    })

@login_required
def create_job(request):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    if request.method == 'POST':
        title = request.POST.get('title')
        company = request.POST.get('company')
        location = request.POST.get('location')
        salary = request.POST.get('salary')
        experience = request.POST.get('experience')
        description = request.POST.get('description')
        skills = request.POST.get('skills')
        
        Job.objects.create(
            title=title,
            company=company,
            location=location,
            salary=salary,
            experience=experience,
            description=description,
            required_skills=skills
        )
        messages.success(request, 'Job Posted Successfully')
        return redirect('hr_dashboard')
        
    return render(request, 'create_job.html')

@login_required
def trigger_scraping(request):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    try:
        count = run_scraper(pages=1)
        messages.success(request, f"Scraping Complete! Found {count} new jobs.")
    except Exception as e:
        messages.error(request, f"Scraping Failed: {str(e)}")
        
    return redirect('hr_dashboard')

@login_required
def update_application_status(request, application_id, status):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    application = get_object_or_404(JobApplication, id=application_id)
    if status in ['SHORTLISTED', 'REJECTED']:
        application.status = status
        application.save()
        if status == 'REJECTED':
            messages.error(request, f"Application {status.title()}!")
        else:
            messages.success(request, f"Application {status.title()}!")
        
    return redirect(request.META.get('HTTP_REFERER', 'hr_dashboard'))

@login_required
def all_applicants(request):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    applications = JobApplication.objects.all().select_related('job', 'applicant_profile').order_by('-applied_date')
    
    return render(request, 'all_applicants.html', {'applications': applications})

@login_required
def job_applications(request, job_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    job = get_object_or_404(Job, id=job_id)
    applications = JobApplication.objects.filter(job=job).select_related('applicant_profile').order_by('-match_score', '-applied_date')
    
    return render(request, 'job_applications_list.html', {
        'job': job,
        'applications': applications
    })

@login_required
def delete_job(request, job_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    job = get_object_or_404(Job, id=job_id)
    job.delete()
    messages.success(request, f"Job '{job.title}' has been deleted successfully.")
    return redirect('hr_dashboard')

@login_required
def view_applicant(request, applicant_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    from app.models import Applicant
    applicant = get_object_or_404(Applicant, id=applicant_id)
    
    # Parse actual skills from string if needed, or template handles it
    # Currently stored as string representation of list '[]'
    import ast
    try:
        skills = ast.literal_eval(applicant.actual_skills)
    except:
        skills = []
        
    # Get missing skills if we want to show recommendations even to HR?
    # Maybe just show what they have.
    # We should show Projects and Certifications too.
    
    return render(request, 'applicant_detail.html', {
        'applicant': applicant,
        'skills': skills
    })

def job_list(request):
    jobs = Job.objects.all().order_by('-posted_date')
    base_template = 'base.html'
    
    if request.user.is_authenticated:
        if request.user.is_superuser:
            base_template = 'hr_base.html'
        else:
            base_template = 'candidate_base.html'
    
    if request.user.is_authenticated and not request.user.is_superuser:
        try:
            from app.models import Applicant
            applicant = Applicant.objects.get(user=request.user)
            if applicant.predicted_category:
                # Filter by category
                # Basic logic: Job title contains category keyword
                # Ideally, this should use more advanced matching from prediction.py
                keyword = applicant.predicted_category.split(' ')[0]
                jobs = jobs.filter(title__icontains=keyword)
                
                if not jobs.exists():
                     messages.info(request, f"No specific jobs found matching your profile category ({applicant.predicted_category}). Showing all jobs.")
                     jobs = Job.objects.all().order_by('-posted_date')
            else:
                 messages.info(request, "Please upload a resume to get personalized job recommendations.")
        except Applicant.DoesNotExist:
            pass # Show all jobs if no profile yet
            
    return render(request, 'job_list.html', {'jobs': jobs, 'base_template': base_template})
    jobs = Job.objects.all().order_by('-posted_date')
    
    if request.user.is_authenticated and not request.user.is_superuser:
        try:
            from app.models import Applicant
            applicant = Applicant.objects.get(user=request.user)
            if applicant.predicted_category:
                # Filter by category
                # Basic logic: Job title contains category keyword
                # Ideally, this should use more advanced matching from prediction.py
                keyword = applicant.predicted_category.split(' ')[0]
                jobs = jobs.filter(title__icontains=keyword)
                
                if not jobs.exists():
                     messages.info(request, f"No specific jobs found matching your profile category ({applicant.predicted_category}). Showing all jobs.")
                     jobs = Job.objects.all().order_by('-posted_date')
            else:
                 messages.info(request, "Please upload a resume to get personalized job recommendations.")
        except Applicant.DoesNotExist:
            pass # Show all jobs if no profile yet
            
    return render(request, 'job_list.html', {'jobs': jobs})
