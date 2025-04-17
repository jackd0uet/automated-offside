import numpy as np
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

class VisualizationHelper():
    '''
    This class acts as an interface for the Sports SoccerPitch rendering.
    '''
    def __init__(self, conf_filter, key_points):
        self.config = SoccerPitchConfiguration()

        # Use keypoint Detection objects and a confidence filter to get points on the pitch.
        self.frame_reference_points = key_points.xy[0][conf_filter]

        # Apply that same filter to the available pitch reference points to have a 1:1 map of points.
        self.pitch_reference_points = np.array(self.config.vertices)[conf_filter]

        self.transformer = self.__set_transformer()

    def __set_transformer(self):
        return ViewTransformer(
            source=self.frame_reference_points,
            target=self.pitch_reference_points
        )

    def transform_points(self, detections):
        ''' Call this three times (ball, players, ref) '''
        frame_xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return self.transformer.transform_points(points=frame_xy)
