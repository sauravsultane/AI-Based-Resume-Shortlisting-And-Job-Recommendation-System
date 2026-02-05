from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('hr-login/', views.hr_login_view, name='hr_login'),
    path('hr-register/', views.hr_register_view, name='hr_register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('resume/upload/', views.upload_resume, name='upload_resume'),
    path('result/', views.result_view, name='result'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    path('add-project/', views.add_project, name='add_project'),
    path('add-certification/', views.add_certification, name='add_certification'),
    path('my-applications/', views.my_applications, name='my_applications'),
    path('my-profile/', views.my_profile, name='my_profile'),
    path('apply/<int:job_id>/', views.apply_job, name='apply_job'),
    path('project/edit/<int:project_id>/', views.edit_project, name='edit_project'),
    path('project/delete/<int:project_id>/', views.delete_project, name='delete_project'),
    path('certification/edit/<int:cert_id>/', views.edit_certification, name='edit_certification'),
    path('certification/delete/<int:cert_id>/', views.delete_certification, name='delete_certification'),
]
