from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import cv2
import numpy as np
import os
import supervision as sv

from .algorithm.classification_helper import ClassificationHelper
from .algorithm.key_point_detection import KeyPointDetection
from .algorithm.object_detection import ObjectDetection
from .algorithm.visualization_helper import VisualizationHelper

base_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(base_dir, 'algorithm', 'weights')

def image_upload_view(request):
    return render(request, "image_upload.html")

@csrf_exempt  # Disable CSRF for simplicity; consider adding proper authentication
def process_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
    
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image file provided"}, status=400)
    
    # Read image from request
    image_file = request.FILES["image"].read()
    np_image = np.frombuffer(image_file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    if image is None:
        return JsonResponse({"error": "Invalid image file"}, status=400)
    
    # Initialize models
    object_detection = ObjectDetection(weights_directory=weights_dir)
    classification_helper = ClassificationHelper()
    key_point_detection = KeyPointDetection()
    
    # Run object detection
    person_detections, ball_detections = object_detection.detect_all(image)
    goalkeepers_detections, players_detections, referees_detections = (
        object_detection.split_detections(person_detections=person_detections)
    )
    
    # Resolve class IDs for all person detections
    player_crops = [sv.crop_image(image, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = classification_helper.team_classifier(player_crops=player_crops)
    
    goalkeepers_detections.class_id = classification_helper.resolve_goalkeepers_team_id(
        players=players_detections, goalkeepers=goalkeepers_detections
    )
    
    referees_detections.class_id -= 1
    players_detections = sv.Detections.merge([players_detections, goalkeepers_detections])
    
    # Key point detection and visualization
    conf_filter, key_points = key_point_detection.detect(image=image)
    visualization = VisualizationHelper(conf_filter=conf_filter, key_points=key_points)
    
    ball_xy = visualization.transform_points(ball_detections)
    players_xy = visualization.transform_points(players_detections)
    refs_xy = visualization.transform_points(referees_detections)
    
    radar_view = visualization.render_pitch(
        ball_xy=ball_xy, players_xy=players_xy, refs_xy=refs_xy, players_detections=players_detections
    )
    
    # Convert image to JPEG format for response
    _, buffer = cv2.imencode(".jpg", radar_view)

    return HttpResponse(buffer.tobytes(), content_type="image/jpeg")
