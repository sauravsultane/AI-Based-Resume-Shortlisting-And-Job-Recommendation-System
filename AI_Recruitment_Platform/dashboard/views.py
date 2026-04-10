import csv
from django.http import HttpResponse
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
def edit_job(request, job_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    job = get_object_or_404(Job, id=job_id)
    
    if request.method == 'POST':
        job.title = request.POST.get('title')
        job.company = request.POST.get('company')
        job.location = request.POST.get('location')
        job.salary = request.POST.get('salary')
        job.experience = request.POST.get('experience')
        job.description = request.POST.get('description')
        job.required_skills = request.POST.get('skills')
        job.save()
        
        messages.success(request, 'Job Updated Successfully')
        return redirect('hr_dashboard')
        
    return render(request, 'create_job.html', {'job': job, 'is_edit': True})

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
    
    # Get sort parameter from URL (default: resume_score)
    sort_by = request.GET.get('sort', 'resume_score')
    
    # Valid sort options
    valid_sorts = {
        'resume_score': '-applicant_profile__resume_score',  # AI Resume Quality Score
        'match_score': '-match_score',  # Job-Resume Match Score
        'date': '-applied_date',
        'name': 'applicant_profile__first_name',
    }
    
    order_field = valid_sorts.get(sort_by, '-applicant_profile__resume_score')
    
    applications = JobApplication.objects.all().select_related(
        'job', 'applicant_profile'
    ).order_by(order_field, '-applied_date')
    
    return render(request, 'all_applicants.html', {
        'applications': applications,
        'current_sort': sort_by
    })

@login_required
def job_applications(request, job_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
    
    job = get_object_or_404(Job, id=job_id)
    
    # Get sort parameter from URL (default: resume_score for AI-powered ranking)
    sort_by = request.GET.get('sort', 'resume_score')
    
    valid_sorts = {
        'resume_score': '-applicant_profile__resume_score',  # AI Resume Quality Score
        'match_score': '-match_score',  # Job-Resume Match Score  
        'date': '-applied_date',
    }
    
    order_field = valid_sorts.get(sort_by, '-applicant_profile__resume_score')
    
    applications = JobApplication.objects.filter(job=job).select_related(
        'applicant_profile'
    ).order_by(order_field, '-applied_date')
    
    return render(request, 'job_applications_list.html', {
        'job': job,
        'applications': applications,
        'current_sort': sort_by
    })

@login_required
def ai_shortlist_job(request, job_id):
    if not request.user.is_superuser:
        return redirect('dashboard')
        
    job = get_object_or_404(Job, id=job_id)
    
    applications_to_shortlist = JobApplication.objects.filter(job=job, match_score__gte=70, status='PENDING')
    count = applications_to_shortlist.update(status='SHORTLISTED')
    
    messages.success(request, f"AI Auto-Shortlisted {count} candidates with match score > 70%!")
    return redirect('job_applications', job_id=job.id)


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
    applied_job_ids = set()
    no_jobs_found = False
    missing_skills = []
    candidate_category = None
    candidate_skills = []

    if request.user.is_authenticated:
        if request.user.is_superuser:
            base_template = 'hr_base.html'
        else:
            base_template = 'candidate_base.html'
            applied_job_ids = set(
                JobApplication.objects.filter(user=request.user)
                .values_list('job_id', flat=True)
            )
            # Only show non-applied jobs
            jobs = jobs.exclude(id__in=applied_job_ids)

    if request.user.is_authenticated and not request.user.is_superuser:
        try:
            from app.models import Applicant
            import ast
            from app.prediction import resume_parser

            applicant = Applicant.objects.get(user=request.user)
            candidate_category = applicant.predicted_category

            if candidate_category:
                keyword = candidate_category.split(' ')[0]
                jobs = jobs.filter(title__icontains=keyword)

                if not jobs.exists():
                    # No matching jobs — show skill recommendations instead of all jobs
                    no_jobs_found = True
                    jobs = Job.objects.none()  # empty queryset

                    # Get skills the candidate is missing for their category
                    resume_text = applicant.resume_text or applicant.actual_skills
                    missing_skills = resume_parser.get_missing_skills(candidate_category, resume_text) or []

                    # Get candidate's current skills for display
                    try:
                        candidate_skills = ast.literal_eval(applicant.actual_skills)
                    except (ValueError, SyntaxError):
                        candidate_skills = []
            else:
                messages.info(request, "Please upload a resume to get personalized job recommendations.")
        except Applicant.DoesNotExist:
            pass

    return render(request, 'job_list.html', {
        'jobs': jobs,
        'base_template': base_template,
        'applied_job_ids': applied_job_ids,
        'no_jobs_found': no_jobs_found,
        'missing_skills': missing_skills,
        'candidate_category': candidate_category,
        'candidate_skills': candidate_skills,
    })

@login_required
def export_csv(request):
    """Export all applicants as a CSV file, sorted by match score (top candidates first)."""
    if not request.user.is_superuser:
        return redirect('dashboard')

    # Get filter param — optionally export only shortlisted
    status_filter = request.GET.get('status', 'ALL')  # ALL | SHORTLISTED | PENDING | REJECTED

    applications = JobApplication.objects.all().select_related(
        'job', 'applicant_profile', 'user'
    ).order_by('-match_score', '-applicant_profile__resume_score')

    if status_filter != 'ALL':
        applications = applications.filter(status=status_filter)

    # Build HTTP response with CSV content type
    response = HttpResponse(content_type='text/csv')
    filename = f'applicants_{status_filter.lower()}.csv'
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)

    # Header row
    writer.writerow([
        'Position',
        'Candidate Name',
        'Email',
        'Job Applied For',
        'Company',
        'Date Applied',
        'Resume Score (/100)',
        'Match Score (%)',
        'Status',
        'Experience Level',
        'Predicted Category',
        'Detected Skills',
    ])

    # Data rows
    for rank, app in enumerate(applications, start=1):
        profile = app.applicant_profile
        if profile:
            import ast
            try:
                skills = ast.literal_eval(profile.actual_skills)
                skills_str = ', '.join(skills)
            except (ValueError, SyntaxError):
                skills_str = profile.actual_skills or ''

            writer.writerow([
                rank,
                f"{profile.first_name} {profile.last_name}",
                app.user.email,
                app.job.title,
                app.job.company,
                app.applied_date.strftime('%b %d, %Y'),
                profile.resume_score,
                f"{app.match_score:.1f}",
                app.get_status_display(),
                profile.experience_level or 'N/A',
                profile.predicted_category or 'N/A',
                skills_str,
            ])
        else:
            writer.writerow([
                rank, 'N/A', app.user.email,
                app.job.title, app.job.company,
                app.applied_date.strftime('%b %d, %Y'),
                'N/A', f"{app.match_score:.1f}",
                app.get_status_display(),
                'N/A', 'N/A', '',
            ])

    return response

