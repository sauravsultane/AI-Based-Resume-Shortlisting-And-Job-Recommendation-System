import os
import ast
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from .forms import UserRegistrationForm, ResumeUploadForm
from .models import Applicant, Project, Certification
from .prediction import resume_parser
import PyPDF2

def index(request):
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('hr_dashboard')
    return render(request, 'index.html')

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, 'Registration successful. Please login.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    # Standard Login Logic
    from django.contrib.auth.forms import AuthenticationForm
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                # Redirect based on user type
                if user.is_superuser:
                    return redirect('hr_dashboard')
                return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'log_in.html', {'form': form})

def hr_login_view(request):
    from django.contrib.auth.forms import AuthenticationForm
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_superuser:
                    login(request, user)
                    return redirect('hr_dashboard')
                else:
                    messages.error(request, "Access Denied: You are not an HR Administrator.")
            else:
                messages.error(request, "Invalid credentials.")
    else:
        form = AuthenticationForm()
    return render(request, 'hr_login.html', {'form': form})

def hr_register_view(request):
    from .forms import HRRegistrationForm
    if request.method == 'POST':
        form = HRRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.is_superuser = True # Make them Admin
            user.is_staff = True
            user.save()
            messages.success(request, 'HR Account Created! Please Login.')
            return redirect('hr_login')
    else:
        form = HRRegistrationForm()
    return render(request, 'hr_register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('index')

@login_required
def dashboard(request):
    # HR users should not access candidate dashboard
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    # Candidate Dashboard
    try:
        applicant = Applicant.objects.filter(user=request.user).last()
    except Applicant.DoesNotExist:
        applicant = None
        
    suggestions = []
    missing_skills = []
    skills_list = []
    job_skill_gaps = []  # New: Job-specific skill gaps
    
    if applicant and applicant.resume_file:
        # Parse skills string to list
        try:
            skills_list = ast.literal_eval(applicant.actual_skills)
        except (ValueError, SyntaxError):
            skills_list = []

        # Use cached resume text — no PDF re-read needed
        text = applicant.resume_text or applicant.actual_skills

        suggestions = resume_parser.generate_suggestions(text, applicant.resume_score)
        missing_skills = resume_parser.get_missing_skills(applicant.predicted_category, text)
        
        # NEW: Calculate job-specific skill gaps from recent applications
        from dashboard.models import JobApplication
        import re
        
        # Helper function to normalize skill names for comparison
        def normalize_skill(skill):
            """Normalize skill name: lowercase, remove spaces/hyphens/dots"""
            return re.sub(r'[\s\-\.\+]+', '', skill.lower().strip())
        
        # Normalize candidate's skills
        skills_normalized = [normalize_skill(s) for s in skills_list]
        
        recent_applications = JobApplication.objects.filter(
            user=request.user
        ).select_related('job').order_by('-applied_date')[:5]  # Last 5 applications
        
        for application in recent_applications:
            job = application.job
            # Parse job required skills (comma-separated)
            required_skills = [s.strip() for s in job.required_skills.split(',') if s.strip()]
            
            # Find missing skills (normalized comparison)
            missing = []
            for skill in required_skills:
                # Compare normalized versions
                if normalize_skill(skill) not in skills_normalized:
                    missing.append(skill)
            
            if missing:
                job_skill_gaps.append({
                    'job_title': job.title,
                    'company': job.company,
                    'missing_skills': missing[:5],  # Limit to top 5 per job
                    'match_score': application.match_score,
                    'total_required': len(required_skills),
                    'total_missing': len(missing)
                })

    return render(request, 'dashboard.html', {
        'applicant': applicant,
        'suggestions': suggestions,
        'missing_skills': missing_skills,
        'skills': skills_list,
        'job_skill_gaps': job_skill_gaps  # New context variable
    })

@login_required
def upload_resume(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            resume_file = request.FILES['resume_file']
            
            # Save Initial Applicant Object
            applicant, created = Applicant.objects.get_or_create(user=request.user)
            applicant.resume_file = resume_file
            applicant.first_name = request.user.first_name
            applicant.last_name = request.user.last_name
            applicant.save()
            
            # --- AI Processing ---
            # 1. Read PDF
            try:
                full_path = os.path.join(settings.MEDIA_ROOT, applicant.resume_file.name)
                text = ""
                page_count = 0

                with open(full_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    page_count = len(pdf_reader.pages)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                # 2. Extract & Predict
                category = resume_parser.predict_category(text)
                extracted_skills = resume_parser.extract_skills(text)
                score = resume_parser.calculate_score(text, page_count)
                level = resume_parser.estimate_experience_level(page_count, text, score)

                # Auto-fill Contact Info
                contact_info = resume_parser.extract_contact_info(text)
                if contact_info['phone'] and not applicant.phone:
                    applicant.phone = contact_info['phone']

                # Education & Location
                edu = resume_parser.extract_education(text)
                loc = resume_parser.extract_location(text)

                if edu['degree']: applicant.education_degree = edu['degree']
                if edu['institution']: applicant.education_institution = edu['institution']
                if loc and not applicant.location: applicant.location = loc

                # 3. Update Model with Multi-Label Prediction
                multi_pred = resume_parser.predict_categories_multi(text, threshold=0.3)
                applicant.predicted_category = multi_pred['primary_category']
                applicant.category_scores = {cat['category']: cat['confidence'] for cat in multi_pred['all_categories']}
                applicant.actual_skills = str(extracted_skills)
                applicant.resume_score = score
                applicant.page_count = page_count
                applicant.experience_level = level
                # Cache raw text so other views don't need to re-read the PDF
                applicant.resume_text = text
                applicant.save()
                
                category_tags = ", ".join(multi_pred['tags'])
                messages.success(request, f"Resume Analyzed! Categories: {category_tags}")
                return redirect('dashboard')
                
            except Exception as e:
                messages.error(request, f"Error processing resume: {str(e)}")
                return redirect('upload_resume')
            
    else:
        form = ResumeUploadForm()
    return render(request, 'resume.html', {'form': form})

@login_required
def result_view(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    applicant = get_object_or_404(Applicant, user=request.user)

    from dashboard.models import Job
    recommended_jobs = Job.objects.filter(
        title__icontains=applicant.predicted_category.split(' ')[0]
    )[:5]

    # Use ast.literal_eval — safe alternative to eval()
    try:
        skills = ast.literal_eval(applicant.actual_skills) if applicant.actual_skills else []
    except (ValueError, SyntaxError):
        skills = []

    context = {
        'applicant': applicant,
        'jobs': recommended_jobs,
        'skills': skills
    }
    return render(request, 'result.html', context)

@login_required
def apply_job(request, job_id):
    if request.user.is_superuser:
        messages.error(request, "HR accounts cannot apply for jobs.")
        return redirect('hr_dashboard')
    from dashboard.models import Job, JobApplication

    job = get_object_or_404(Job, id=job_id)
    applicant = get_object_or_404(Applicant, user=request.user)
    
    # Check if already applied
    if JobApplication.objects.filter(user=request.user, job=job).exists():
        messages.warning(request, f"You have already applied for {job.title} at {job.company}.")
    else:
        # Use cached resume text — no need to re-read the PDF file
        full_text = applicant.resume_text or applicant.actual_skills
            
        from app.prediction import resume_parser
        match_percentage = resume_parser.calculate_match_percentage(
            job_description=job.description,
            job_skills=job.required_skills,
            resume_text=full_text
        )
        
        JobApplication.objects.create(
            user=request.user,
            job=job,
            applicant_profile=applicant,
            status='PENDING',
            match_score=match_percentage
        )
        messages.success(request, f"Applied! Your Profile Match: {match_percentage}%")
        
    return redirect('job_list')

@login_required
def edit_profile(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from .forms import ProfileUpdateForm, ApplicantProfileForm

    applicant, created = Applicant.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        user_form = ProfileUpdateForm(request.POST, instance=request.user)
        profile_form = ApplicantProfileForm(request.POST, request.FILES, instance=applicant)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            
            # Sync Name
            applicant.first_name = request.user.first_name
            applicant.last_name = request.user.last_name
            applicant.save()
            
            # Re-trigger AI if a new resume was uploaded
            if 'resume_file' in request.FILES:
                try:
                    full_path = os.path.join(settings.MEDIA_ROOT, applicant.resume_file.name)
                    text = ""
                    with open(full_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        page_count = len(pdf_reader.pages)
                        for page in pdf_reader.pages:
                            text += page.extract_text()

                    category = resume_parser.predict_category(text)
                    extracted_skills = resume_parser.extract_skills(text)
                    score = resume_parser.calculate_score(text, page_count)
                    level = resume_parser.estimate_experience_level(page_count, text, score)

                    applicant.predicted_category = category
                    applicant.actual_skills = str(extracted_skills)
                    applicant.resume_score = score
                    applicant.page_count = page_count
                    applicant.experience_level = level
                    # Update cached text so subsequent views stay consistent
                    applicant.resume_text = text
                    applicant.save()
                    messages.info(request, "Resume re-analyzed successfully!")
                except Exception as e:
                    messages.warning(request, f"Resume uploaded but analysis failed: {e}")

            messages.success(request, 'Profile Updated Successfully')
            return redirect('dashboard')
    else:
        user_form = ProfileUpdateForm(instance=request.user)
        profile_form = ApplicantProfileForm(instance=applicant)
    
    return render(request, 'edit_profile.html', {
        'user_form': user_form, 
        'profile_form': profile_form,
        'applicant': applicant
    })

@login_required
def add_project(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from .forms import ProjectForm
    applicant = get_object_or_404(Applicant, user=request.user)
    
    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.applicant = applicant
            project.save()
            messages.success(request, 'Project Added!')
            return redirect('edit_profile')
    else:
        form = ProjectForm()
    return render(request, 'add_item.html', {'form': form, 'title': 'Add Project'})

@login_required
def add_certification(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from .forms import CertificationForm
    applicant = get_object_or_404(Applicant, user=request.user)
    
    if request.method == 'POST':
        form = CertificationForm(request.POST)
        if form.is_valid():
            cert = form.save(commit=False)
            cert.applicant = applicant
            cert.save()
            messages.success(request, 'Certification Added!')
            return redirect('edit_profile')
    else:
        form = CertificationForm()
    return render(request, 'add_item.html', {'form': form, 'title': 'Add Certification'})

@login_required
def my_applications(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from dashboard.models import JobApplication
    applications = JobApplication.objects.filter(user=request.user).select_related('job').order_by('-applied_date')
    
    return render(request, 'my_applications_list.html', {'applications': applications})

@login_required
def my_profile(request):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    applicant = get_object_or_404(Applicant, user=request.user)

    import ast
    try:
        skills = ast.literal_eval(applicant.actual_skills)
    except:
        skills = []
        
    return render(request, 'applicant_detail.html', {
        'applicant': applicant,
        'skills': skills,
        'base_template': 'candidate_base.html'
    })

@login_required
def edit_project(request, project_id):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from .forms import ProjectForm
    project = get_object_or_404(Project, id=project_id, applicant__user=request.user)
    
    if request.method == 'POST':
        form = ProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            messages.success(request, 'Project Updated!')
            return redirect('edit_profile')
    else:
        form = ProjectForm(instance=project)
    return render(request, 'add_item.html', {'form': form, 'title': 'Edit Project'})

@login_required
def delete_project(request, project_id):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    project = get_object_or_404(Project, id=project_id, applicant__user=request.user)
    project.delete()
    messages.success(request, 'Project Deleted!')
    return redirect('edit_profile')

@login_required
def edit_certification(request, cert_id):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    from .forms import CertificationForm
    cert = get_object_or_404(Certification, id=cert_id, applicant__user=request.user)
    
    if request.method == 'POST':
        form = CertificationForm(request.POST, instance=cert)
        if form.is_valid():
            form.save()
            messages.success(request, 'Certification Updated!')
            return redirect('edit_profile')
    else:
        form = CertificationForm(instance=cert)
    return render(request, 'add_item.html', {'form': form, 'title': 'Edit Certification'})

@login_required
def delete_certification(request, cert_id):
    if request.user.is_superuser:
        return redirect('hr_dashboard')
    cert = get_object_or_404(Certification, id=cert_id, applicant__user=request.user)
    cert.delete()
    messages.success(request, 'Certification Deleted!')
    return redirect('edit_profile')
