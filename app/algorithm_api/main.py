import base64
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import json
import logging
import numpy as np
import os
from pathlib import Path
import shutil
import supervision as sv
import traceback

from algorithm.classification_helper import ClassificationHelper
from algorithm.key_point_detection import KeyPointDetection
from algorithm.object_detection import ObjectDetection
from algorithm.offside_classification import OffsideClassification
from algorithm.visualization_helper import VisualizationHelper

from utils import convert_to_serializable

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

object_detection = ObjectDetection(weights_directory="weights")
key_point_detection = KeyPointDetection()

@app.get("/")
def read_root():
    return {"message": "Algorithm API is running"}

@app.post("/object-detection/")
async def detection(
    image: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    classification_helper = ClassificationHelper()

    if confidence != 0.5:
        object_detection.threshold = confidence

    try:
        file_location = UPLOAD_DIR / image.filename

        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)
        decoded_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if decoded_image is None:
            raise ValueError("Image decoding failed, invalid format or corrupted image.")

        # Object detection
        person_detections, ball_detections = object_detection.detect_all(image=decoded_image)
        goalkeepers_detections, players_detections, referees_detections = object_detection.split_detections(person_detections)

        # Classification
        player_crops = [sv.crop_image(decoded_image, xyxy) for xyxy in players_detections.xyxy]

        players_detections.class_id = classification_helper.team_classifier(player_crops)

        goalkeepers_detections.class_id = classification_helper.resolve_goalkeepers_team_id(
            players=players_detections, goalkeepers=goalkeepers_detections
        )

        players_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

        referees_detections.class_id -= 1

        # Field detection & visualization
        conf_filter, key_points = key_point_detection.detect(image=decoded_image)
        visualization = VisualizationHelper(conf_filter=conf_filter, key_points=key_points)

        ball_xy = visualization.transform_points(ball_detections)
        players_xy = visualization.transform_points(players_detections)
        refs_xy = visualization.transform_points(referees_detections)

        try:
            image.file.seek(0)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        except Exception as e:
            logging.warning(f"Image saving failed: {traceback.format_exc()}")

        # Return processed data
        # TODO: use convert_to_serializable here
        response = {
            'ball_xy': {
                "tracker_id": ball_detections.tracker_id.tolist(),
                "xy": ball_xy.tolist(),
            },
            'players_xy': {
                "tracker_id": players_detections.tracker_id.tolist(),
                "xy": players_xy.tolist(),
            },
            'refs_xy': {
                "tracker_id" : referees_detections.tracker_id.tolist(),
                "xy": refs_xy.tolist(),
            },
            'players_detections': {
                "xyxy": players_detections.xyxy.tolist(),
                "confidence": players_detections.confidence.tolist(),
                "class_id": players_detections.class_id.tolist(),
                "tracker_id": players_detections.tracker_id.tolist(),
                "class_name": players_detections.data["class_name"].tolist(),
            },
            'file_path' : str(file_location)
        }

        return JSONResponse(content=response)
    
    except Exception as e:
        logging.error(f"Error during image processing: {traceback.format_exc()}")
        return JSONResponse({"error": f"Failed to process image: {str(e)}"}, status_code=500)

@app.post("/offside-classification/")
async def offside_classification(request: Request):
    try:
        data = await request.body()
        data = json.loads(data.decode("utf-8"))

        players_detections = {
            'xyxy': np.array(data['players_detections']['xyxy']),
            'confidence': np.array(data['players_detections']['confidence']),
            'class_id': np.array(data['players_detections']['class_id']),
            'tracker_id': np.array(data['players_detections']['tracker_id']),
            'class_name': np.array(data['players_detections']['class_name'], dtype=str),
        }

        classification_helper = OffsideClassification(players_detections)
        offside_status, second_defender = classification_helper.classify()

        return JSONResponse(content=convert_to_serializable({
            'offside_status': offside_status,
            'second_defender': {
                'tracker_id': second_defender,
            }
        }))
    
    except Exception as e:
        logging.error(f"Error during offside classification: {traceback.format_exc()}")
        return JSONResponse({"error": f"Failed to determine offside: {str(e)}"}, status_code=500)

@app.post("/image_picker/")
async def pick_image(request: Request):
    try:
        data = await request.json()
        file_path = data['file_path'].strip('"')

        BASE_DIR = os.path.dirname(__file__)
        abs_path = os.path.join(BASE_DIR, file_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        image = cv2.imread(abs_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {abs_path}")

        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            'image': image_base64
        })

    except Exception as e:
        logging.error(f"Error during image retrieval: {traceback.format_exc()}")
        return JSONResponse({"error": f"Failed to retrieve image: {str(e)}"}, status_code=500)
