from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.db.models import F
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.utils.timezone import now

import base64
import cv2
from datetime import datetime, timedelta
import json
import logging
import requests
import traceback

from .models import OffsideDecision, ObjectDetection
from .utils import *

logging = logging.getLogger(__name__)

def index(request):
    decisions = OffsideDecision.objects.all()
    total = decisions.count()
    correct = decisions.filter(algorithm_decision=F("final_decision")).count()
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    recent_upload = decisions.order_by('-time_uploaded').first()

    context = {
        'total_decisions': total,
        'accuracy': accuracy,
        'recent_upload': recent_upload.time_uploaded if recent_upload else None,
        'year': now().year,
    }

    return render(request, "index.html", context)

def login_view(request):
    if request.method == "POST":

        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect(request.POST.get("next") or "index")
    else:
        form = AuthenticationForm(request)

    return render(request, "login.html", {"form": form, "next": request.GET.get("next", "")})

@login_required
def upload_image(request):
    return render(request, "image_upload.html")

@login_required
def logs_view(request):
    offside_decisions = OffsideDecision.objects.all().order_by('-time_uploaded')

    preset = request.GET.get("preset")
    start_date = request.GET.get("start_date")
    end_date = request.GET.get("end_date")

    today = now()

    if preset == "last_week":
        start = today - timedelta(days=7)
        offside_decisions = offside_decisions.filter(time_uploaded__date__gte=start)
    elif preset == "last_month":
        start = today - timedelta(days=30)
        offside_decisions = offside_decisions.filter(time_uploaded__date__gte=start)
    elif preset == "last_year":
        start = today - timedelta(days=365)
        offside_decisions = offside_decisions.filter(time_uploaded__date__gte=start)

    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            offside_decisions = offside_decisions.filter(time_uploaded__date__gte=start)
        except ValueError:
            pass

    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            offside_decisions = offside_decisions.filter(time_uploaded__date__lte=end)
        except ValueError:
            pass

    total = offside_decisions.count()
    correct = offside_decisions.filter(algorithm_decision=F("final_decision")).count()
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    context = {
        'offside_decisions': offside_decisions,
        'accuracy': accuracy,
    }

    return render(request, "logs.html", context)

def object_detection_detail(request, id, time_uploaded):
    detection = get_object_or_404(ObjectDetection, id=id)

    data = {
        'players_detections': json.loads(detection.players_detections),
        'players_xy': json.loads(detection.players_xy),
        'ball_xy': json.loads(detection.ball_xy),
        'refs_xy': json.loads(detection.refs_xy)
    }

    ball_xy, players_xy, refs_xy, players_detections = format_json(data)

    image = render_pitch(ball_xy, players_xy, refs_xy, players_detections)
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    context = {
        'detection_image': encoded_image,
        'detection_time': time_uploaded
    }

    return render(request, 'object_detection_detail.html', context)

def process_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        request.session['time_uploaded'] = str(now())
        image_file = request.FILES['image']
        confidence = request.POST.get('confidence', 0.5)

        url = "http://127.0.0.1:8002/object-detection/"
        files = {'image': image_file}
        data = {'confidence' : confidence}
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            response_data = response.json()

            ObjectDetection.objects.create(
                players_detections=json.dumps(response_data['players_detections']),
                players_xy=json.dumps(response_data['players_xy']),
                ball_xy=json.dumps(response_data['ball_xy']),
                refs_xy=json.dumps(response_data['refs_xy']),
                file_path=json.dumps(response_data['file_path'])
            )

            request.session['object_detection_id'] = ObjectDetection.objects.latest('id').id

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

            ball_xy, players_xy, refs_xy, players_detections = format_json(data)

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
    try:
        classification_result = request.session.get('classification_result', None)
        detection_data = request.session.get('POST_data')['detection_data']

        ball_xy, players_xy, refs_xy, _ = format_json(detection_data)

        image = render_offside(
            ball_xy,
            players_xy,
            refs_xy,
            classification_result
        )
        
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        request.session['offside_radar_view'] = encoded_image
    
    except Exception as e:
        logging.error(f"Error rendering offside: {traceback.format_exc()}")
        return JsonResponse({'error': f"Failed to render offside: {str(e)}"}, status=500)

@login_required
def display_offside(request):
    classification_result = request.session.get('classification_result', None)
    offside_radar_view = request.session.get('offside_radar_view', None)

    offside_count = sum(1 for player in classification_result.values() if player['offside'])

    algorithm_decision = "Offside" if offside_count > 0 else "Onside"

    return render(
        request,
        'offside_decision.html',
        {
            'classification_result': classification_result,
            'algorithm_decision': algorithm_decision,
            'offside_radar_view': offside_radar_view
        }
    )

def store_offside(request):
    if request.method == "POST":
        try:
            decision_time = str(now())
            time_uploaded = request.session.get('time_uploaded')

            detection_id = request.session.get('object_detection_id')
            detection = ObjectDetection.objects.get(id=detection_id)

            referee = request.user

            data = json.loads(request.body.decode("utf-8"))

            algorithm_decision = data['algorithm_decision']
            final_decision = data['final_decision']

            OffsideDecision.objects.create(
                detection_id=detection,
                referee_id=referee,
                algorithm_decision=algorithm_decision,
                final_decision=final_decision,
                time_uploaded=time_uploaded,
                time_decided=decision_time
            )
            decision_id = OffsideDecision.objects.latest('id').id

            logging.warning(f"Store offside successful, with id: {decision_id}")

            return JsonResponse({'decision_id': decision_id}, status=200)

        except Exception as e:
            logging.error(f"Error saving decision: {traceback.format_exc()}")
            return JsonResponse({'error': f"Failed to save decision: {str(e)}"}, status=500)

def update_detections(request):
    if request.method == "POST":
        try:
            detection_id = request.session.get('object_detection_id')

            request_data = request.body.decode('utf-8')
            request_data = request_data.replace("'", '"')
            new_data = json.loads(request_data)

            players_detections = new_data['players_detections']
            players_xy = new_data['players_xy']

            detection = ObjectDetection.objects.get(id=detection_id)
            detection.players_detections = json.dumps(players_detections)
            detection.players_xy = json.dumps(players_xy)
            detection.save()

            return JsonResponse({'success': f"Detection: {detection_id} successfully updated!"}, status=200)

        except Exception as e:
            logging.error(f"Failed to update detection:{detection_id}, reason: {traceback.format_exc()}")
            return JsonResponse({'error': f"Couldn't update detection:{detection_id}: {str(e)}"})
