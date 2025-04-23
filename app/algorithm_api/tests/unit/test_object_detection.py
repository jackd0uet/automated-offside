import unittest
import cv2

from algorithm.object_detection import ObjectDetection

class TestObjectDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = ObjectDetection(weights_directory='./weights', threshold=0.5)
        cls.image = cv2.imread('./tests/integration/images/221_jpg.rf.a5b76a00596073c23f1254a62e945536.jpg')

    def test_detect_all(self):
        person_detections, ball_detections = self.detector.detect_all(self.image)

        self.assertIsNotNone(person_detections)
        self.assertIsNotNone(ball_detections)

        self.assertTrue(hasattr(person_detections, "xyxy"))
        self.assertTrue(hasattr(ball_detections, "xyxy"))

    def test_split_detections(self):
        person_detections, _ = self.detector.detect_all(self.image)
        gk, players, refs = self.detector.split_detections(person_detections)

        self.assertTrue(hasattr(gk, "xyxy"))
        self.assertTrue(hasattr(players, "xyxy"))
        self.assertTrue(hasattr(refs, "xyxy"))