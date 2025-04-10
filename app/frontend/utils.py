import cv2
import numpy as np
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

config = SoccerPitchConfiguration()

def draw_legend(image, legend):
    # TODO: move legend to other side if players are taking up the space
    x_start, y_start = 300, 100
    spacing =  50
    radius = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    for i, (label, colour) in enumerate(legend):
        centre = (x_start, y_start + i * spacing)
        cv2.circle(image, centre, radius, colour.as_bgr(), -1)
        cv2.putText(
            image,
            label,
            (centre[0] + 20, centre[1] + 7),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA
        )

    return image

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

    legend = [
        ('Ball', sv.Color.WHITE),
        ('Team A', sv.Color.from_hex('00BFFF')),
        ('Team B', sv.Color.from_hex('FF1493')),
        ('Referee', sv.Color.from_hex('FFD700')),
    ]

    annotated_image = draw_legend(annotated_image, legend)

    return annotated_image

def render_offside(ball_xy, players_xy, refs_xy, classification_result):
    defenders_xy, offside_xy, onside_xy = get_plottables(classification_result, players_xy)
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
        xy=defenders_xy,
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=offside_xy,
        face_color=sv.Color.from_hex('FF0000'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image
    )
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=onside_xy,
        face_color=sv.Color.from_hex('00FF00'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image
    )

    annotated_image = draw_points_on_pitch(
        config=config,
        xy=refs_xy['xy'],
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    legend = [
        ('Ball', sv.Color.WHITE),
        ('Defending Team', sv.Color.from_hex('00BFFF')),
        ('Attacking Team (OFFSIDE)', sv.Color.from_hex('FF0000')),
        ('Attacking Team (ONSIDE)', sv.Color.from_hex('00FF00')),
        ('Referee', sv.Color.from_hex('FFD700')),
    ]

    annotated_image = draw_legend(annotated_image, legend)

    return annotated_image


def get_plottables(classification_result, players_xy):
    # Get tracker IDs for attacking players
    remove_ids = {player['tracker_id'] for player in classification_result.values()}

    tracker_ids = players_xy['tracker_id']
    xy_coords = players_xy['xy']

    # Split XYs into defenders and attackers based on tracker IDs
    defenders_xy = xy_coords[~np.isin(tracker_ids, list(remove_ids))]
    attackers_xy = xy_coords[np.isin(tracker_ids, list(remove_ids))]

    # Get the attackers tracker IDs
    attackers_tracker_ids = tracker_ids[np.isin(tracker_ids, list(remove_ids))]

    # Find the players who are considered offside and create a mask for them
    offside_lookup = {player['tracker_id']: player['offside'] for player in classification_result.values()}
    offside_mask = np.array([offside_lookup.get(tid, False) for tid in attackers_tracker_ids])

    # Split attackers into onside and offside for plotting
    offside_xy = attackers_xy[offside_mask]
    onside_xy = attackers_xy[~offside_mask]

    return defenders_xy, offside_xy, onside_xy

def format_json(data):
    ball_xy = {
                'tracker_id': np.array(data['ball_xy']['tracker_id']),
                'xy': np.array(data['ball_xy']['xy']),
            }

    players_xy = {
        'tracker_id': np.array(data['players_xy']['tracker_id']),
        'xy': np.array(data['players_xy']['xy']),
    }

    refs_xy = {
        'tracker_id': np.array(data['refs_xy']['tracker_id']),
        'xy': np.array(data['refs_xy']['xy']),
    }

    players_detections = {
        'xyxy': np.array(data['players_detections']['xyxy']),
        'confidence': np.array(data['players_detections']['confidence']),
        'class_id': np.array(data['players_detections']['class_id']),
        'tracker_id': np.array(data['players_detections']['tracker_id']),
        'class_name': np.array(data['players_detections']['class_name'])
    }

    return ball_xy, players_xy, refs_xy, players_detections
