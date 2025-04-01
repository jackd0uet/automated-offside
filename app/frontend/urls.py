from django.urls import path
from .views import index, login, upload_image, process_image, render_pitch_view

urlpatterns = [
    path("", index, name="index"),
    path("login/", login, name="login"),
    path("upload/", upload_image, name="upload_image"),
    path("process_image/", process_image, name="process_image"),
    path("render_pitch/", render_pitch_view, name="render_pitch")
]