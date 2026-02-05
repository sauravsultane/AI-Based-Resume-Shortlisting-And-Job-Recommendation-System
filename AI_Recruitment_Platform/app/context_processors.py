from .models import Applicant

def applicant_context(request):
    if request.user.is_authenticated:
        # Avoid error for superusers or users without applicant profile
        try:
            applicant = Applicant.objects.filter(user=request.user).last()
            return {'global_applicant': applicant}
        except:
            pass
    return {}
