from inference import get_model
import os
import supervision as sv

class KeyPointDetection():
    def __init__(self, model_id=None):
        self.token = os.environ.get("ROBOFLOW_API_KEY")
        self.model_id = "football-field-detection-f07vi/14" if model_id == None else model_id

        if self.token and self.model_id:
            self.model = self.load_model()

    def load_model(self):
        return get_model(model_id=self.model_id, api_key=self.token)
    
    def detect(self, image, confidence=0.5):
        if not hasattr(self, 'model'):
            raise AttributeError("The KeyPointDetection model has not been loaded correctly.")         

        result = self.model.infer(image)[0]

        key_points = sv.KeyPoints.from_inference(result)

        conf_filter = key_points.confidence[0] > confidence

        return conf_filter, key_points
