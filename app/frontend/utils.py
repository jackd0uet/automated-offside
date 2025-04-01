from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

def render_pitch(ball_xy, players_xy, refs_xy, players_detections):
    config = SoccerPitchConfiguration()

    annotated_image = draw_pitch(config)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_image
    )

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=players_xy[players_detections['class_id'] == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=players_xy[players_detections['class_id'] == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=refs_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    return annotated_image
