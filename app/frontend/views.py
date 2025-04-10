from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.urls import reverse

import base64
import cv2
import json
import logging
import numpy as np
import requests
import traceback

from .utils import render_offside, render_pitch

logging = logging.getLogger(__name__)

def index(request):
    return render(request, "index.html")

def login(request):
    return render(request, "login.html")

def upload_image(request):
    return render(request, "image_upload.html")

def process_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES['image']
        confidence = 0.5

        if request.FILES.get("confidence"):
            confidence = request.POST['confidence']

        url = "http://127.0.0.1:8002/object-detection/"
        files = {'image': image_file}
        data = {'confidence' : confidence}
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            return HttpResponse(response.content)
        else:
            return JsonResponse({
                "error": "Failed to process image",
                "details": response.text
            }, status=500)
        
    return JsonResponse({"error": "No image provided"}, status=400)

def render_pitch_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))

            ball_xy = {
                'tracker_id': np.array(data['ball_xy']['tracker_id']),
                'xy': np.array(data['ball_xy']['xy']),
            }

            players_xy = {
                'tracker_id': np.array(data['players_xy']['tracker_id']),
                'xy': np.array(data['players_xy']['xy']),
            }

            refs_xy = {
                'tracker_id': np.array(data['refs_xy']['tracker_id']),
                'xy': np.array(data['refs_xy']['xy']),
            }

            players_detections = {
                'xyxy': np.array(data['players_detections']['xyxy']),
                'confidence': np.array(data['players_detections']['confidence']),
                'class_id': np.array(data['players_detections']['class_id']),
                'tracker_id': np.array(data['players_detections']['tracker_id']),
                'class_name': np.array(data['players_detections']['class_name'])
            }

            image = render_pitch(ball_xy, players_xy, refs_xy, players_detections)

            _, buffer = cv2.imencode('.jpg', image)
            return HttpResponse(buffer.tobytes(), content_type='image/jpeg')

        except Exception as e:
            logging.error(f"Error rendering pitch: {traceback.format_exc()}")
            return JsonResponse({'error': str(e)}, status=500)

def classify_offside(request):
    if request.method == "POST":
        url = "http://127.0.0.1:8002/offside-classification/"

        try:
            payload = json.loads(request.body)

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                classification_json = response.json()

                request.session['POST_data'] = payload

                request.session['classification_result'] = classification_json['offside_status']
                request.session['second_defender'] = classification_json['second_defender']

                render_offside_view(request=request)

                return JsonResponse({
                    "redirect_url": reverse('display_offside')
                })

            else:
                return JsonResponse({
                    "error": "Failed to determine offside classification",
                    "details": response.text
                }, status=500)

        except json.JSONDecodeError:
            return JsonResponse({'error': "Invalid JSON in request body"}, status=400)
        except Exception as e:
            logging.error(f"Something went wrong: {traceback.format_exc()}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': "Only POST requests to this endpoint are permitted"}, status=400)

def render_offside_view(request):
    classification_result = request.session.get('classification_result', None)
    second_defender = request.session.get('second_defender', None)
    data = request.session.get('POST_data')

    try:
        image = render_offside(
            data=data,
            classification_result=classification_result,
            second_defender=second_defender)
        
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        request.session['offside_radar_view'] = encoded_image
    
    except Exception as e:
        logging.error(f"Error rendering offside: {traceback.format_exc()}")
        return JsonResponse({'error': f"Failed to render offside: {str(e)}"}, status=500)

def display_offside(request):
    classification_result = request.session.get('classification_result', None)
    offside_radar_view = request.session.get('offside_radar_view', None)

    return render(
        request,
        'offside_decision.html',
        {
            'classification_result': classification_result,
            'offside_radar_view': offside_radar_view
        }
    )
    