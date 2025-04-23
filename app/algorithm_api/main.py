import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Request
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
from algorithm.visualisation_helper import VisualisationHelper

from utils import convert_to_serializable

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialise these outside of calls to the API, improves speed of calls.
object_detection = ObjectDetection(weights_directory="weights")
key_point_detection = KeyPointDetection()

@app.get("/")
def read_root():
    return {"message": "Algorithm API is running"}

@app.post("/object-detection/")
async def detection(request: Request):
    try:
        # At the moment form is only confidence but could accept more user control later.
        form = await request.form()

        # Instantiate classification helper now, there were some weird persistence issues if started earlier.
        classification_helper = ClassificationHelper()

        # Handle confidence value.
        confidence_str = form.get("confidence", "0.5")
        try:
            confidence = float(confidence_str)
        except ValueError:
            return JSONResponse(content={"error": "Invalid confidence value"}, status_code=400)

        # Set confidence if user has inputted a new one.
        if confidence != 0.5:
            object_detection.threshold = confidence
            key_point_detection.confidence = confidence

        image: UploadFile = form.get("image")

        if image is None:
            return JSONResponse(content={"error": "Image is required"}, status_code=400)

        # Prep for image storage.
        file_location = UPLOAD_DIR / image.filename

        # Image pre-processing
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

        # Field detection & visualisation
        conf_filter, key_points = key_point_detection.detect(image=decoded_image)
        visualisation = VisualisationHelper(conf_filter=conf_filter, key_points=key_points)

        ball_xy = visualisation.transform_points(ball_detections)
        players_xy = visualisation.transform_points(players_detections)
        refs_xy = visualisation.transform_points(referees_detections)

        # Save uploaded image for future reference.
        # Note: this stores locally currently but could be stored on a server in future.
        try:
            image.file.seek(0)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        except Exception as e:
            logging.warning(f"Image saving failed: {traceback.format_exc()}")

        # Return processed data
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
        # Get data passed into POST request.
        data = await request.body()
        data = json.loads(data.decode("utf-8"))

        # Split data.
        players_detections_data = data['detection_data']['players_detections']
        defending_team = data['defending_team']

        # Assemble detections object.
        players_detections = {
            'xyxy': np.array(players_detections_data['xyxy']),
            'confidence': np.array(players_detections_data['confidence']),
            'class_id': np.array(players_detections_data['class_id']),
            'tracker_id': np.array(players_detections_data['tracker_id']),
            'class_name': np.array(players_detections_data['class_name'], dtype=str),
        }

        # Run detections through Offside Classification.
        classification_helper = OffsideClassification(players_detections, defending_team)
        offside_status, second_defender = classification_helper.classify()

        # Return object with offside status for each player and the tracker id for the second defender.
        return JSONResponse(content=convert_to_serializable({
            'offside_status': offside_status,
            'second_defender': {
                'tracker_id': second_defender,
            }
        }))
    
    except Exception as e:
        logging.error(f"Error during offside classification: {traceback.format_exc()}")
        return JSONResponse({"error": f"Failed to determine offside: {str(e)}"}, status_code=500)
