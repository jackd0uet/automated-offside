import unittest
from unittest.mock import patch
import numpy as np
import supervision as sv

class MockSoccerPitchConfiguration:
    def __init__(self):
        self.length = 100
        self.width = 70

SoccerPitchConfiguration = MockSoccerPitchConfiguration

from frontend.utils import (
    draw_legend,
    draw_labels_on_pitch,
    render_pitch,
    render_offside,
    get_plottables,
    format_json,
)

class TestAnnotationFunctions(unittest.TestCase):

    def test_draw_legend_left_orientation(self):
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        legend_items = [('Label 1', sv.Color.RED), ('Label 2', sv.Color.GREEN)]

        result_image = draw_legend(image.copy(), legend_items, orientation='left')

        self.assertIsNotNone(result_image)
        self.assertEqual(image.shape, result_image.shape)

    def test_draw_legend_right_orientation(self):
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        legend_items = [('Label A', sv.Color.BLUE), ('Label B', sv.Color.YELLOW)]

        result_image = draw_legend(image.copy(), legend_items, orientation='right')

        self.assertIsNotNone(result_image)
        self.assertEqual(image.shape, result_image.shape)

    def test_draw_labels_on_pitch(self):
        pitch = np.zeros((200, 300, 3), dtype=np.uint8)
        xy = np.array([[50, 50], [150, 100]])
        labels = ['Player 1', 'Player 2']

        result_pitch = draw_labels_on_pitch(pitch.copy(), xy, labels)

        self.assertIsNotNone(result_pitch)
        self.assertEqual(pitch.shape, result_pitch.shape)

    @patch('frontend.utils.draw_pitch')
    @patch('frontend.utils.draw_points_on_pitch')
    @patch('frontend.utils.draw_labels_on_pitch')
    @patch('frontend.utils.draw_legend')
    def test_render_pitch(self, mock_draw_legend, mock_draw_labels, mock_draw_points, mock_draw_pitch):
        mock_draw_pitch.return_value = np.zeros((680, 1050, 3), dtype=np.uint8)
        ball_xy = {'xy': np.array([[525, 340]])}
        players_xy = {'xy': np.array([[100, 100], [900, 500], [200, 400], [800, 200]]), 'tracker_id': np.array([1, 2, 3, 4])}
        refs_xy = {'xy': np.array([[500, 300]])}
        players_detections = {
            'class_id': np.array([0, 1, 0, 1]),
            'tracker_id': np.array([1, 2, 3, 4])
        }

        result_image = render_pitch(ball_xy, players_xy, refs_xy, players_detections)

        self.assertIsNotNone(result_image)
        mock_draw_pitch.assert_called_once()
        self.assertEqual(mock_draw_points.call_count, 4) # Ball, Team A, Team B, Referees
        self.assertEqual(mock_draw_labels.call_count, 2) # Team A, Team B
        mock_draw_legend.assert_called_once()

    def test_get_plottables(self):
        classification_result = {
            0: {'tracker_id': 1, 'offside': True},
            1: {'tracker_id': 3, 'offside': False}
        }
        players_xy = {
            'tracker_id': np.array([0, 1, 2, 3]),
            'xy': np.array([[50, 50], [100, 100], [150, 150], [200, 200]])
        }

        defenders_xy, offside_xy, onside_xy = get_plottables(classification_result, players_xy)

        np.testing.assert_array_equal(defenders_xy, np.array([[50, 50], [150, 150]]))
        np.testing.assert_array_equal(offside_xy, np.array([[100, 100]]))
        np.testing.assert_array_equal(onside_xy, np.array([[200, 200]]))

    @patch('frontend.utils.draw_pitch')
    @patch('frontend.utils.draw_points_on_pitch')
    @patch('frontend.utils.draw_legend')
    @patch('frontend.utils.get_plottables')
    def test_render_offside(self, mock_get_plottables, mock_draw_legend, mock_draw_points, mock_draw_pitch):
        mock_draw_pitch.return_value = np.zeros((680, 1050, 3), dtype=np.uint8)
        mock_get_plottables.return_value = (
            np.array([[100, 100]]),  # defenders_xy
            np.array([[200, 200]]),  # offside_xy
            np.array([[300, 300]])   # onside_xy
        )

        ball_xy = {'xy': np.array([[525, 340]])}
        players_xy = {'xy': np.array([[100, 100], [200, 200], [300, 300], [400, 400]]), 'tracker_id': np.array([1, 2, 3, 4])}
        refs_xy = {'xy': np.array([[500, 300]])}
        classification_result = {0: {'tracker_id': 2, 'offside': True}}

        result_image = render_offside(ball_xy, players_xy, refs_xy, classification_result)

        self.assertIsNotNone(result_image)
        mock_draw_pitch.assert_called_once()
        self.assertEqual(mock_draw_points.call_count, 5) # Ball, Defenders, Offside, Onside, Referees
        mock_draw_legend.assert_called_once()
        mock_get_plottables.assert_called_once_with(classification_result, players_xy)

    def test_format_json_with_all_data(self):
        data = {
            'ball_xy': {'tracker_id': [99], 'xy': [[500, 300]]},
            'players_xy': {'tracker_id': [1, 2], 'xy': [[100, 100], [200, 200]]},
            'refs_xy': {'tracker_id': [10, 11], 'xy': [[300, 150], [400, 250]]},
            'players_detections': {
                'xyxy': [[50, 50, 70, 70], [150, 150, 170, 170]],
                'confidence': [0.9, 0.8],
                'class_id': [0, 1],
                'tracker_id': [1, 2],
                'class_name': ['team_a', 'team_b']
            }
        }
        ball_xy, players_xy, refs_xy, players_detections = format_json(data)

        np.testing.assert_array_equal(ball_xy['xy'], np.array([[500, 300]]))
        np.testing.assert_array_equal(players_xy['xy'], np.array([[100, 100], [200, 200]]))
        np.testing.assert_array_equal(refs_xy['xy'], np.array([[300, 150], [400, 250]]))
        np.testing.assert_array_equal(players_detections['class_id'], np.array([0, 1]))

    def test_format_json_without_refs_and_ball(self):
        data = {
            'players_xy': {'tracker_id': [1, 2], 'xy': [[100, 100], [200, 200]]},
            'players_detections': {
                'xyxy': [[50, 50, 70, 70], [150, 150, 170, 170]],
                'confidence': [0.9, 0.8],
                'class_id': [0, 1],
                'tracker_id': [1, 2],
                'class_name': ['team_a', 'team_b']
            }
        }
        ball_xy, players_xy, refs_xy, players_detections = format_json(data)

        self.assertEqual(ball_xy, {})
        self.assertEqual(refs_xy, {})
        np.testing.assert_array_equal(players_xy['xy'], np.array([[100, 100], [200, 200]]))
        np.testing.assert_array_equal(players_detections['class_id'], np.array([0, 1]))
