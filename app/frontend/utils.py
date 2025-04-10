from sports.annotators.soccer import (
    draw_paths_on_pitch,
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

config = SoccerPitchConfiguration()

def render_pitch(ball_xy, players_xy, refs_xy, players_detections):
    annotated_image = draw_pitch(config)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=ball_xy['xy'],
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_image
    )

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=players_xy['xy'][players_detections['class_id'] == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=players_xy['xy'][players_detections['class_id'] == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=refs_xy['xy'],
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    return annotated_image

def render_offside(xyxy, classification_result, second_defender):
    offside_line = [[second_defender[0], config.width], [second_defender[0], 0]]

    annotated_image = render_pitch(
        xyxy['ball_xy'],
        xyxy['players_xy'],
        xyxy['refs_xy'],
        xyxy['players_detections']
    )

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=classification_result[["xyxy"]["offside"] == True],
        face_color=sv.Color.from_hex("FF0000"),
        edge_color=sv.Color.WHITE,
        radius=16,
        pitch=annotated_image
    )
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=classification_result[["xyxy"]["offside"] == False],
        face_color=sv.Color.from_hex("00FF00"),
        edge_color=sv.Color.WHITE,
        radius=16,
        pitch=annotated_image
    )

    annotated_image = draw_paths_on_pitch(
        config=config,
        paths=offside_line,
        pitch=annotated_image
    )

    return annotated_image
