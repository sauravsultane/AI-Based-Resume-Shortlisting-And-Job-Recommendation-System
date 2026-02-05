from django.contrib import admin
from .models import Job, JobApplication

class JobApplicationAdmin(admin.ModelAdmin):
    list_display = ('user', 'job', 'status', 'match_score', 'applied_date')
    list_filter = ('status', 'job')

admin.site.register(Job)
admin.site.register(JobApplication, JobApplicationAdmin)
