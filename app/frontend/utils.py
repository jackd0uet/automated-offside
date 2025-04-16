import cv2
import numpy as np
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

config = SoccerPitchConfiguration()

def draw_legend(image, legend, orientation='left'):
    # TODO: move legend to other side if players are taking up the space
    x_start, y_start = 300, 100

    if orientation == 'right':
        x_start = image.shape[1] - 300

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

def draw_labels_on_pitch(pitch, xy, labels, padding=50, scale=0.1):
    for point, label in zip(xy, labels):
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = text_size

        text_x = scaled_point[0] - text_width // 2
        text_y = scaled_point[1] + text_height // 2

        cv2.putText(
            img=pitch,
            text=label,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    return pitch

def render_pitch(ball_xy, players_xy, refs_xy, players_detections):
    annotated_image = draw_pitch(config)

    # Draw ball
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=ball_xy['xy'],
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_image
    )

    # Team A
    team_a_mask = players_detections['class_id'] == 0
    team_a_xy = players_xy['xy'][team_a_mask]
    team_a_labels = [str(tid) for tid in players_detections['tracker_id'][team_a_mask]]
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=team_a_xy,
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)
    
    annotated_image = draw_labels_on_pitch(
        pitch=annotated_image,
        xy=team_a_xy,
        labels=team_a_labels
    )

    # Team B
    team_b_mask = players_detections['class_id'] == 1
    team_b_xy = players_xy['xy'][team_b_mask]
    team_b_labels = [str(tid) for tid in players_detections['tracker_id'][team_b_mask]]
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=team_b_xy,
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    annotated_image = draw_labels_on_pitch(
        pitch=annotated_image,
        xy=team_b_xy,
        labels=team_b_labels
    )

    # Referees
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=refs_xy['xy'],
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image)

    # Determine which side the legend should be on
    avg_x = np.mean(players_xy['xy'][:, 0])

    orientation = 'right' if avg_x < config.length / 2 else 'left'

    # Legend
    legend = [
        ('Ball', sv.Color.WHITE),
        ('Team A', sv.Color.from_hex('00BFFF')),
        ('Team B', sv.Color.from_hex('FF1493')),
        ('Referee', sv.Color.from_hex('FFD700')),
    ]
    annotated_image = draw_legend(annotated_image, legend, orientation)

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

    # Determine which side the legend should be on
    avg_x = np.mean(players_xy['xy'][:, 0])

    orientation = 'right' if avg_x < config.length / 2 else 'left'

    legend = [
        ('Ball', sv.Color.WHITE),
        ('Defending Team', sv.Color.from_hex('00BFFF')),
        ('Attacking Team (OFFSIDE)', sv.Color.from_hex('FF0000')),
        ('Attacking Team (ONSIDE)', sv.Color.from_hex('00FF00')),
        ('Referee', sv.Color.from_hex('FFD700')),
    ]

    annotated_image = draw_legend(annotated_image, legend, orientation)

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
    ball_xy = {}
    refs_xy = {}

    if 'ball_xy' in data:
        ball_xy = {
                    'tracker_id': np.array(data['ball_xy']['tracker_id']),
                    'xy': np.array(data['ball_xy']['xy']),
            }

    players_xy = {
        'tracker_id': np.array(data['players_xy']['tracker_id']),
        'xy': np.array(data['players_xy']['xy']),
    }

    if 'refs_xy' in data:
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
