from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import cv2
import json
import logging
import numpy as np
import requests

from .utils import render_pitch

logging = logging.getLogger(__name__)

def index(request):
    return render(request, "index.html")

def login(request):
    return render(request, "login.html")

def upload_image(request):
    return render(request, "image_upload.html")

@csrf_exempt
def process_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        url = "http://127.0.0.1:8002/object_detection/"
        files = {'image': image_file}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            return HttpResponse(response.content, content_type="image/jpeg")
        else:
            return JsonResponse({"error": "Failed to process image"}, status=500)
        
    return JsonResponse({"error": "No image provided"}, status=400)

@csrf_exempt
def render_pitch_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))

            ball_xy = np.array(data["ball_xy"])
            players_xy = np.array(data["players_xy"])
            refs_xy = np.array(data["refs_xy"])
            players_detections = {
                'xyxy': np.array(data['players_detections']['xyxy']),
                'confidence': np.array(data['players_detections']['confidence']),
                'class_id': np.array(data['players_detections']['class_id']),
                'class_name': np.array(data['players_detections']['class_name'])
            }

            image = render_pitch(ball_xy, players_xy, refs_xy, players_detections)

            _, buffer = cv2.imencode('.jpg', image)
            return HttpResponse(buffer.tobytes(), content_type="image/jpeg")

        except Exception as e:
            logging.error(f"Error rendering pitch: {e}")
            return JsonResponse({"error": str(e)}, status=500)

def classify_offside(request):
    # TODO: implement call to classify_offside endpoint
    pass
