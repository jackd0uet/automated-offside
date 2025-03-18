import numpy as np
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

class VisualizationHelper():
    def __init__(self, conf_filter, key_points):
        self.config = SoccerPitchConfiguration()

        self.frame_reference_points = key_points.xy[0][conf_filter]
        self.pitch_reference_points = np.array(self.config.vertices)[conf_filter]

        self.transformer = self.set_transformer()

    def set_transformer(self):
        return ViewTransformer(
            source=self.frame_reference_points,
            target=self.pitch_reference_points
        )

    # call this three times (ball, players, ref)
    def transform_points(self, detections):
        frame_xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return self.transformer.transform_points(points=frame_xy)
    
    def render_pitch(self, ball_xy, players_xy, refs_xy, players_detections):
        annotated_image = draw_pitch(self.config)

        annotated_image = draw_points_on_pitch(
            config=self.config,
            xy=ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=annotated_image
        )

        annotated_image = draw_points_on_pitch(
            config=self.config,
            xy=players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_image)

        annotated_image = draw_points_on_pitch(
            config=self.config,
            xy=players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_image)

        annotated_image = draw_points_on_pitch(
            config=self.config,
            xy=refs_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_image)

        return annotated_image
