from inference import get_model
import os
import supervision as sv

class KeyPointDetection():
    '''
    This class uses the football field detection model trained by Roboflow to map a football pitch.
    This class also allows users to input their own model which could be used in the future.
    football-field-detection model: https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi
    '''
    def __init__(self, model_id=None, confidence = 0.5):
        self.token = os.environ.get("ROBOFLOW_API_KEY")
        self.model_id = "football-field-detection-f07vi/14" if model_id == None else model_id

        # Setting confidence here allows users to rerun detections if insufficient keypoints are detected.
        self.confidence = confidence

        if self.token and self.model_id:
            self.model = get_model(model_id=self.model_id, api_key=self.token)
    
    def detect(self, image):
        if not hasattr(self, 'model'):
            raise AttributeError("The KeyPointDetection model has not been loaded correctly.")         

        # Use the models infer function to generate keypoints data.
        result = self.model.infer(image)[0]

        # Convert the keypoints into supervision format for later use.
        key_points = sv.KeyPoints.from_inference(result)

        # Create a confidence filter so that only points above a chosen confidence threshold are used.
        conf_filter = key_points.confidence[0] > self.confidence

        return conf_filter, key_points
