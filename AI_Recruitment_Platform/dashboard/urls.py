from django.urls import path
from . import views

urlpatterns = [
    path('hr/', views.hr_dashboard, name='hr_dashboard'),
    path('post-job/', views.create_job, name='create_job'),
    path('scrape/', views.trigger_scraping, name='trigger_scraping'),
    path('update-status/<int:application_id>/<str:status>/', views.update_application_status, name='update_status'),
    path('all-applicants/', views.all_applicants, name='all_applicants'),
    path('job/<int:job_id>/applications/', views.job_applications, name='job_applications'),
    path('delete-job/<int:job_id>/', views.delete_job, name='delete_job'),
    path('applicant/<int:applicant_id>/', views.view_applicant, name='view_applicant'),
    path('jobs/', views.job_list, name='job_list'),
]
