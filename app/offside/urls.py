"""
URL configuration for offside project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("", include("frontend.urls")),
    path("admin/", admin.site.urls),
    path("decision/", include("decision.urls")),
]
