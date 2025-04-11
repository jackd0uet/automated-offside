from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from django.contrib.auth.models import User

import datetime
from unittest.mock import patch, MagicMock
from io import BytesIO
import json
import logging
import numpy as np
from PIL import Image

from frontend.models import ObjectDetection, OffsideDecision

def get_test_image():
    file = BytesIO()
    image = Image.new('RGB', (100, 100))
    image.save(file, 'jpeg')
    file.name = 'test.jpg'
    file.seek(0)
    return file

class ViewsTestCase(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_index_view(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)

    def test_login_view_get(self):
        self.client.logout()
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

    def test_login_view_post_invalid(self):
        self.client.logout()
        response = self.client.post(reverse('login'), {
            'username': 'invalid',
            'password': 'invalid'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "form")

    def test_upload_image_view(self):
        response = self.client.get(reverse('upload_image'))
        self.assertEqual(response.status_code, 200)

    def test_logs_view_requires_login(self):
        response = self.client.get(reverse('logs'))
        self.assertEqual(response.status_code, 200)

    def test_object_detection_detail(self):
        detection = ObjectDetection.objects.create(
            players_detections="[]", players_xy="[]", ball_xy="[]",
            refs_xy="[]", file_path="[]"
        )
        response = self.client.get(reverse('object_detection_detail', args=[detection.id, 'now']))
        self.assertEqual(response.status_code, 200)

    @patch('requests.post')
    def test_process_image_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'players_detections': [],
            'players_xy': [],
            'ball_xy': [],
            'refs_xy': [],
            'file_path': ''
        }
        mock_post.return_value = mock_response

        image = get_test_image()
        response = self.client.post(reverse('process_image'), {'image': image})
        self.assertEqual(response.status_code, 200)

    def test_process_image_no_image(self):
        response = self.client.post(reverse('process_image'), {})
        self.assertEqual(response.status_code, 400)

    @patch('cv2.imencode')
    @patch('frontend.views.render_pitch')
    def test_render_pitch_view_success(self, mock_render_pitch, mock_imencode):
        mock_render_pitch.return_value = MagicMock()
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        data = {
            'ball_xy': {'tracker_id': [], 'xy': []},
            'players_xy': {'tracker_id': [], 'xy': []},
            'refs_xy': {'tracker_id': [], 'xy': []},
            'players_detections': {
                'xyxy': [],
                'confidence': [],
                'class_name': [],
                'class_id': [],
                'tracker_id': []
            }
        }

        response = self.client.post(
            reverse('render_pitch'),
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)

    @patch('requests.post')
    def test_classify_offside_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'offside_status': {'1': {'offside': True}},
                'second_defender': [1, 2]
            }
        )

        data = {'dummy': 'data'}
        response = self.client.post(reverse('classify_offside'), data=json.dumps(data),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('redirect_url', response.json())

    def test_classify_offside_invalid_json(self):
        response = self.client.post(reverse('classify_offside'), data="{invalid_json}",
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_display_offside_view(self):
        session = self.client.session
        session['classification_result'] = {'1': {'offside': True}, '2': {'offside': False}}
        session['offside_radar_view'] = 'fake_base64'
        session.save()

        response = self.client.get(reverse('display_offside'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Offside')

    def test_store_offside_success(self):
        detection = ObjectDetection.objects.create(
            players_detections='[]',
            players_xy='[]',
            ball_xy='[]',
            refs_xy='[]',
            file_path='[]'
        )

        session = self.client.session
        session['object_detection_id'] = detection.id
        session['time_uploaded'] = str(timezone.make_aware(datetime.datetime.now()))
        session.save()

        data = {
            'algorithm_decision': 'Offside',
            'final_decision': 'Onside'
        }

        response = self.client.post(
            reverse('store_offside'),
            data=json.dumps(data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('success', response.json())
