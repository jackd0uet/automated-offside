import unittest
from unittest.mock import patch, MagicMock
import os

from algorithm.key_point_detection import KeyPointDetection

class TestKeyPointsDetection(unittest.TestCase):

    @patch.dict(os.environ, {'ROBOFLOW_API_KEY': 'fake_api_key'})
    @patch('algorithm.key_point_detection.get_model')
    def setUp(self, mock_get_model):
        self.mock_model = MagicMock()

        mock_get_model.return_value = self.mock_model

        self.mock_model.infer.return_value = [{
            'keypoints': [[100, 200], [150, 250]],
            'confidence': [0.9, 0.85]
        }]

        self.keypoint_detector = KeyPointDetection(model_id="football-field-detection-f07vi/14", confidence=0.8)

    def test_initialisation_default_model(self):
        detector = KeyPointDetection(confidence=0.7)
        self.assertEqual(detector.model_id, "football-field-detection-f07vi/14")

    @patch('algorithm.key_point_detection.get_model')
    def test_initialization_custom_model(self, mock_get_model):
        mock_get_model.return_value = MagicMock()

        detector = KeyPointDetection(model_id="custom_model_id", confidence=0.7)
        self.assertEqual(detector.model_id, "custom_model_id")

    @patch('algorithm.key_point_detection.sv.KeyPoints')
    def test_detect(self, mock_keypoints_class):
        mock_keypoints = MagicMock()
        mock_keypoints.confidence = [0.9, 0.85]
        mock_keypoints_class.from_inference.return_value = mock_keypoints

        conf_filter, key_points = self.keypoint_detector.detect('dummy_image.jpg')

        self.assertEqual(conf_filter, True)
        self.assertEqual(len(key_points.confidence), 2)
        self.assertEqual(key_points.confidence[0], 0.9)
    
    def test_detect_model_not_loaded(self):
        detector = KeyPointDetection(confidence=0.7)

        with self.assertRaises(AttributeError):
            detector.dummy('dummy_image.jpg')
