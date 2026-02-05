from django import forms
from django.contrib.auth.models import User
from .models import Applicant

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")
        return cleaned_data

class HRRegistrationForm(UserRegistrationForm):
    secret_code = forms.CharField(max_length=20, required=True, help_text="Enter HR Security Code")
    
    def clean_secret_code(self):
        code = self.cleaned_data.get('secret_code')
        if code != 'HR2026':  # Simple hardcoded secret for testing
            raise forms.ValidationError("Invalid HR Security Code")
        return code

class ResumeUploadForm(forms.ModelForm):
    class Meta:
        model = Applicant
        fields = ['resume_file']

class ProfileUpdateForm(forms.ModelForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class ApplicantProfileForm(forms.ModelForm):
    class Meta:
        model = Applicant
        fields = ['phone', 'location', 'gender', 'languages', 'education_degree', 'education_institution', 'resume_file']
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'resume_file': forms.FileInput(attrs={'class': 'form-control'}),
        }

class ProjectForm(forms.ModelForm):
    class Meta:
        from .models import Project
        model = Project
        fields = ['title', 'link', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class CertificationForm(forms.ModelForm):
    class Meta:
        from .models import Certification
        model = Certification
        fields = ['title', 'link', 'issue_date']
        widgets = {
            'issue_date': forms.DateInput(attrs={'type': 'date'}),
        }
