import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
import logging
import numpy as np
import os
import supervision as sv

from algorithm.classification_helper import ClassificationHelper
from algorithm.key_point_detection import KeyPointDetection
from algorithm.object_detection import ObjectDetection
from algorithm.offside_classification import OffsideClassification
from algorithm.visualization_helper import VisualizationHelper

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI()

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
        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image decoding failed, invalid format or corrupted image.")

        # Object detection
        person_detections, ball_detections = object_detection.detect_all(image=image)
        goalkeepers_detections, players_detections, referees_detections = object_detection.split_detections(person_detections)

        # Classification
        player_crops = [sv.crop_image(image, xyxy) for xyxy in players_detections.xyxy]

        players_detections.class_id = classification_helper.team_classifier(player_crops)

        goalkeepers_detections.class_id = classification_helper.resolve_goalkeepers_team_id(
            players=players_detections, goalkeepers=goalkeepers_detections
        )

        players_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

        referees_detections.class_id -= 1

        # Field detection & visualization
        conf_filter, key_points = key_point_detection.detect(image=image)
        visualization = VisualizationHelper(conf_filter=conf_filter, key_points=key_points)

        ball_xy = visualization.transform_points(ball_detections)
        players_xy = visualization.transform_points(players_detections)
        refs_xy = visualization.transform_points(referees_detections)

        # Return processed data
        response = {
            'ball_xy': ball_xy.tolist(),
            'players_xy': players_xy.tolist(),
            'refs_xy': refs_xy.tolist(),
            'players_detections': {
                "xyxy": players_detections.xyxy.tolist(),
                "confidence": players_detections.confidence.tolist(),
                "class_id": players_detections.class_id.tolist(),
                "class_name": players_detections.data["class_name"].tolist(),
            }
        }

        return JSONResponse(content=response)
    
    except Exception as e:
        logging.error(f"Error during image processing: {str(e)}")
        return JSONResponse({"error": f"Failed to process image: {str(e)}"}, status_code=500)

    finally:
        del players_detections
        del goalkeepers_detections
        del referees_detections
