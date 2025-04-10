import numpy as np
from sports.annotators.soccer import (
    draw_paths_on_pitch,
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv

config = SoccerPitchConfiguration()

def render_pitch(ball_xy, players_xy, refs_xy, players_detections):
    # TODO: add legend
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

def render_offside(ball_xy, players_xy, refs_xy, classification_result):
    # TODO: add legend
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
        face_color=sv.Color.from_hex("FF0000"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_image
    )
    annotated_image = draw_points_on_pitch(
        config=config,
        xy=onside_xy,
        face_color=sv.Color.from_hex("00FF00"),
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
