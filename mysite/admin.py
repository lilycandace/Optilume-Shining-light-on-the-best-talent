
from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils.html import format_html
from mysite.models import Contact
from mysite.models import PostJob
from mysite.models import Apply_job


# Register your models here.
@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'subject']
    search_fields = ['name', 'email', 'subject']
    
    actions = ['redirect_to_monitoring', 'redirect_to_main_monitoring']
    
    def redirect_to_monitoring(self, request, queryset):
        """Redirect to Streamlit system monitoring page"""
        return HttpResponseRedirect('http://localhost:8501/?page=graphs')
    
    def redirect_to_main_monitoring(self, request, queryset):
        """Redirect to main Streamlit monitoring page"""
        return HttpResponseRedirect('http://localhost:8501')
    
    redirect_to_monitoring.short_description = "📈 Open System Monitoring Dashboard"
    redirect_to_main_monitoring.short_description = "📊 Open Main Monitoring Dashboard"

@admin.register(Apply_job)
class ApplyJobAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'company_name', 'title', 'experience']
    list_filter = ['gender', 'company_name']
    search_fields = ['name', 'email', 'company_name', 'title']
    
    actions = ['redirect_to_monitoring', 'redirect_to_main_monitoring']
    
    def redirect_to_monitoring(self, request, queryset):
        """Redirect to Streamlit system monitoring page"""
        return HttpResponseRedirect('http://localhost:8501/?page=graphs')
    
    def redirect_to_main_monitoring(self, request, queryset):
        """Redirect to main Streamlit monitoring page"""
        return HttpResponseRedirect('http://localhost:8501')
    
    redirect_to_monitoring.short_description = "📈 Open System Monitoring Dashboard"
    redirect_to_main_monitoring.short_description = "📊 Open Main Monitoring Dashboard"

# Custom admin site header with monitoring link
admin.site.site_header = "Optilume Admin Dashboard"
admin.site.site_title = "Optilume Admin"
admin.site.index_title = format_html(
    'Welcome to Optilume Admin Portal | <a href="http://localhost:8501" target="_blank" style="color: #417690; text-decoration: none; margin-left: 20px; font-size: 16px;">📊 Monitoring</a> | <a href="http://localhost:8501/?page=graphs" target="_blank" style="color: #28a745; text-decoration: none; margin-left: 20px; font-size: 16px;">📈 System Monitoring</a>'
)

# Custom admin action to redirect to monitoring
@admin.register(PostJob)
class PostJobAdmin(admin.ModelAdmin):
    list_display = ['title', 'company_name', 'job_location', 'employment_status']
    list_filter = ['employment_status', 'company_name', 'gender']
    search_fields = ['title', 'company_name', 'details']
    
    actions = ['redirect_to_monitoring', 'redirect_to_main_monitoring']
    
    def redirect_to_monitoring(self, request, queryset):
        """Redirect to Streamlit system monitoring page"""
        return HttpResponseRedirect('http://localhost:8501/?page=graphs')
    
    def redirect_to_main_monitoring(self, request, queryset):
        """Redirect to main Streamlit monitoring page"""
        return HttpResponseRedirect('http://localhost:8501')
    
    redirect_to_monitoring.short_description = "📈 Open System Monitoring Dashboard"
    redirect_to_main_monitoring.short_description = "📊 Open Main Monitoring Dashboard"


