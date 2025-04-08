import supervision as sv
from ultralytics import YOLO

class ObjectDetection():
    def __init__(self, weights_directory, threshold=0.5):
        self.base_model = YOLO(f"{weights_directory}/base/best.pt")
        self.ball_model = YOLO(f"{weights_directory}/ball/best.pt")

        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3

        self.threshold = threshold

    def __base_detector(self, image, model):
        result = model(image)[0]

        return sv.Detections.from_ultralytics(result)

    def detect_all(self, image):
        detections = self.__base_detector(image=image, model=self.base_model)

        if len(detections[detections.class_id == self.BALL_ID]) > 0:
            ball_detections = detections[detections.class_id == self.BALL_ID]
        else:
            ball_detections = self.__base_detector(image=image, model=self.ball_model)

        person_detections = detections[detections.class_id != self.BALL_ID]
        person_detections = person_detections.with_nms(threshold=self.threshold, class_agnostic=True)

        return person_detections, ball_detections
    
    def split_detections(self, person_detections):
        goalkeepers_detections = person_detections[person_detections.class_id == self.GOALKEEPER_ID]
        players_detections = person_detections[person_detections.class_id == self.PLAYER_ID]
        referees_detections = person_detections[person_detections.class_id == self.REFEREE_ID]

        return goalkeepers_detections, players_detections, referees_detections
