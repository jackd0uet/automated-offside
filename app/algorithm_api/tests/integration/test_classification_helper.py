import unittest
import numpy as np
import cv2
import supervision as sv

from algorithm.classification_helper import ClassificationHelper

class DummyDetections:
    def __init__(self, boxes, class_ids):
        self.boxes = boxes  # (x_center, y_center, width, height)
        self.class_id = np.array(class_ids)

    def get_anchors_coordinates(self, position):
        if position == sv.Position.BOTTOM_CENTER:
            return np.array([
                [x, y + h / 2] for (x, y, w, h) in self.boxes
            ])
        raise ValueError("Unsupported position")

class TestClassificationHelperIntegration(unittest.TestCase):

    def setUp(self):
        self.helper = ClassificationHelper()

    def generate_dummy_player_crop(self, color=(255, 0, 0)):
        img = np.full((100, 50, 3), color, dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Test for classifying players into teams
    def test_team_classifier_real_features(self):
        # 4 dummy crops: 2 red, 2 blue to simulate different team colors
        red_players = [self.generate_dummy_player_crop((255, 0, 0)) for _ in range(2)]
        blue_players = [self.generate_dummy_player_crop((0, 0, 255)) for _ in range(2)]
        all_players = red_players + blue_players

        labels = self.helper.team_classifier(all_players)
        self.assertEqual(len(labels), 4)
        self.assertTrue(set(labels).issubset({0, 1}))

    # Test for resolving goalkeepers team
    def test_resolve_goalkeepers_team_id_real_coords(self):
        # Simulate bounding boxes (x, y, w, h)
        player_boxes = [
            (100, 100, 20, 40),  # team 0
            (120, 100, 20, 40),  # team 0
            (300, 300, 20, 40),  # team 1
            (320, 300, 20, 40),  # team 1
        ]
        goalkeeper_boxes = [
            (110, 110, 20, 40),  # closer to team 0
            (310, 310, 20, 40),  # closer to team 1
        ]

        players = DummyDetections(player_boxes, [0, 0, 1, 1])
        goalkeepers = DummyDetections(goalkeeper_boxes, [0, 0])

        result = self.helper.resolve_goalkeepers_team_id(players, goalkeepers)
        self.assertTrue(np.array_equal(result, np.array([0, 1])))