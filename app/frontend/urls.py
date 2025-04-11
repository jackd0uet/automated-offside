from django.contrib.auth.views import LogoutView
from django.urls import path
from .views import index, login_view, upload_image, process_image, render_pitch_view, classify_offside, display_offside, store_offside

urlpatterns = [
    path("", index, name="index"),
    path("login/", login_view, name="login"),
    path("logout/", LogoutView.as_view(next_page="index"), name="logout"),
    path("upload/", upload_image, name="upload_image"),
    path("process_image/", process_image, name="process_image"),
    path("render_pitch/", render_pitch_view, name="render_pitch"),
    path("classify_offside/", classify_offside, name="classify_offside"),
    path("display_offside/", display_offside, name="display_offside"),
    path("store_offside/", store_offside, name="store_offside"),
]
