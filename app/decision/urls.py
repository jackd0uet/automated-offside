from django.urls import path
from .views import process_image, image_upload_view

urlpatterns = [
    path("process-image/", process_image, name="process_image"),
    path("upload-image/", image_upload_view, name="upload_image"),
]
