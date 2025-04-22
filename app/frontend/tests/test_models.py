import unittest

from django.contrib.auth.models import User
from django.utils.timezone import now

from frontend.models import ObjectDetection, OffsideDecision

class TestModels(unittest.TestCase):
    def setUp(self):
        self.file_path = "test/filepath.jpg"
        self.detection = ObjectDetection.objects.create(
            file_path=self.file_path
        )
        self.user, created = User.objects.get_or_create(username='models_tester', defaults={'password': 'password'})
    

    def test_object_detection_string_formatting(self):
        self.assertEqual(f"{self.detection.id} | {self.file_path}", str(self.detection))

    def test_offside_decision_string_formatting(self):
        current = now()
        

        decision = OffsideDecision.objects.create(
            detection_id=self.detection,
            referee_id=self.user,
            algorithm_decision=True,
            final_decision=True,
            time_uploaded=current,
            time_decided=current
        )

        decision_string = f"{decision.id} | {self.detection.id} | True | True | {current} | {current}"

        self.assertEqual(decision_string, str(decision))