import numpy as np
import supervision as sv
from ultralytics import YOLO

class ObjectDetection():
    '''
    This class uses Ultralytic's YOLO 11l implementation as a base model which has been trained on football images.
    '''
    def __init__(self, weights_directory, threshold=0.5):
        # There are two models in play here, one a generic football identifier 
        # and another more specifically for ball detection.
        self.base_model = YOLO(f"{weights_directory}/base/best.pt")
        self.ball_model = YOLO(f"{weights_directory}/ball/best.pt")

        # These IDs have been set as part of the model training.
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3

        # Allows for user confidence threshold setting.
        self.threshold = threshold

    def __base_detector(self, image, model):
        # Generate bounding boxes from the inputted image.
        result = model(image)[0]

        # Convert to Supervision Detections object for future use.
        detections = sv.Detections.from_ultralytics(result)

        # Give each detection a unique tracker id for later use.
        num_detections = detections.xyxy.shape[0]
        unique_ids = np.arange(num_detections)
        detections.tracker_id = unique_ids

        return detections

    def detect_all(self, image):
        # First run the generic detector.
        # This finds balls, goalkeepers, players and referees.
        detections = self.__base_detector(image=image, model=self.base_model)

        # If the generic detector doesn't find a ball
        # then use the model that is trained specifically for ball detection.
        if len(detections[detections.class_id == self.BALL_ID]) > 0:
            ball_detections = detections[detections.class_id == self.BALL_ID]
        else:
            ball_detections = self.__base_detector(image=image, model=self.ball_model)

        # Get all person detections and use non-maximum suppression to select the most promising bounding boxes.
        person_detections = detections[detections.class_id != self.BALL_ID]
        person_detections = person_detections.with_nms(threshold=self.threshold, class_agnostic=True)

        return person_detections, ball_detections
    
    def split_detections(self, person_detections):
        # Split the detections into distinct classes.
        goalkeepers_detections = person_detections[person_detections.class_id == self.GOALKEEPER_ID]
        players_detections = person_detections[person_detections.class_id == self.PLAYER_ID]
        referees_detections = person_detections[person_detections.class_id == self.REFEREE_ID]

        return goalkeepers_detections, players_detections, referees_detections
