from django.test import TestCase, Client, RequestFactory
from django.urls import reverse
from django.utils.timezone import now, make_aware
from django.contrib.auth.models import User

from datetime import timedelta, datetime
from unittest.mock import patch, MagicMock
from io import BytesIO
import json
import logging
import numpy as np
from PIL import Image

from frontend.models import ObjectDetection, OffsideDecision
from frontend.views import render_offside_view

def get_test_image():
    file = BytesIO()
    image = Image.new('RGB', (100, 100))
    image.save(file, 'jpeg')
    file.name = 'test.jpg'
    file.seek(0)
    return file

class IndexViewTestCase(TestCase):
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

class AuthenticationViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_login_view_get(self):
        self.client.logout()
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

    def test_login_view_post_success(self):
        response = self.client.post(reverse('login'), {
            'username': 'tester',
            'password': 'password'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue('_auth_user_id' in self.client.session)

    def test_login_view_post_invalid(self):
        self.client.logout()
        response = self.client.post(reverse('login'), {
            'username': 'invalid',
            'password': 'invalid'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "form")

class UploadImageViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_upload_image_view_success(self):
        response = self.client.get(reverse('upload_image'))
        self.assertEqual(response.status_code, 200)

    def test_upload_image_view_failure(self):
        self.client.logout()

        response = self.client.post(reverse('upload_image'))

        self.assertEqual(response.status_code, 302)

class ObjectDetectionViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_object_detection_detail(self):
        detection = ObjectDetection.objects.create(
            players_detections='{"xyxy": [],"confidence": [],"class_name": [],"class_id": [],"tracker_id": []}',
            players_xy='{"tracker_id": [1], "xy": [[100, 50]]}',
            ball_xy='{"tracker_id": [], "xy": []}',
            refs_xy='{"tracker_id": [], "xy": []}',
            file_path='[]'
        )
        response = self.client.get(reverse('object_detection_detail', args=[detection.id, 'now']))
        self.assertEqual(response.status_code, 200)

class RenderPitchViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

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

    def test_render_pitch_view_failure(self):
        response = self.client.post(reverse('render_pitch'))

        self.assertEqual(response.status_code, 500)

class RenderOffsideViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @patch('cv2.imencode')
    @patch('frontend.views.render_offside')
    def test_render_offside_view_success(self, mock_render_offside, mock_imencode):
        mock_render_offside.return_value = MagicMock()
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        request_factory = RequestFactory()
        request = request_factory.post(reverse('display_offside'))

        request.session = self.client.session
        request.session['POST_data'] = {
            'detection_data': {
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
        }
        request.session.save()

        render_offside_view(request)

        self.assertIsNotNone(request.session['offside_radar_view'])

    def test_render_offside_view_failure(self):
        self.client.logout()

        response = self.client.post(reverse('display_offside'))

        self.assertEqual(response.status_code, 302)

class ClassifyOffsideViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_classify_offside_get_failure(self):
        response = self.client.get(reverse('classify_offside'))
        self.assertEqual(response.status_code, 400)

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

    @patch('requests.post')
    def test_classify_offside_failure(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=500,
            json=lambda: {
                'error': 'Failed to determine offside:'
            }
        )

        data = {'dummy': 'data'}
        response = self.client.post(reverse('classify_offside'), data=json.dumps(data),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 500)

    def test_classify_offside_invalid_json(self):
        response = self.client.post(reverse('classify_offside'), data="{invalid_json}",
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)

class DisplayOffsideViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_display_offside_view(self):
        session = self.client.session
        session['classification_result'] = {'1': {'offside': True}, '2': {'offside': False}}
        session['offside_radar_view'] = 'fake_base64'
        session.save()

        response = self.client.get(reverse('display_offside'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Offside')

class StoreOffsideViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.login(username='tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_store_offside_success(self):
        detection = ObjectDetection.objects.create(
            players_detections='[]',
            players_xy='[]',
            ball_xy='[]',
            refs_xy='[]',
            file_path='[]'
        )
        expected_detection_id = detection.id
        expected_time_uploaded = str(now())
        expected_algo_decision = 'Offside'
        expected_fin_decision = 'Onside'

        session = self.client.session
        session['object_detection_id'] = expected_detection_id
        session['time_uploaded'] = expected_time_uploaded
        session.save()

        data = {
            'algorithm_decision': expected_algo_decision,
            'final_decision': expected_fin_decision,
        }

        response = self.client.post(
            reverse('store_offside'),
            data=json.dumps(data),
            content_type='application/json'
        )

        self.assertEqual(OffsideDecision.objects.count(), 1)
        obj = OffsideDecision.objects.latest('id')

        self.assertEqual(obj.detection_id.id, expected_detection_id)
        self.assertEqual(obj.algorithm_decision, expected_algo_decision)
        self.assertEqual(obj.final_decision, expected_fin_decision)
        self.assertEqual(obj.time_uploaded, datetime.fromisoformat(expected_time_uploaded))

        self.assertEqual(response.status_code, 200)

    def test_store_offside_failure(self):
        response = self.client.post(
            reverse('store_offside'),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.json())
        self.assertEqual(OffsideDecision.objects.count(), 0)

class ProcessImageViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='process_image_tester', password='password')
        self.client.login(username='process_image_tester', password='password')
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @patch('frontend.views.requests.post')
    def test_process_image_success(self, mock_post):
        expected_detections = [{'id': 1}]
        expected_players_xy = [[100, 150]]
        expected_ball_xy = [200, 250]
        expected_refs_xy = [[50, 60]]
        expected_file_path = 'test_path.jpg'

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_response.json.return_value = {
            'players_detections': expected_detections,
            'players_xy': expected_players_xy,
            'ball_xy': expected_ball_xy,
            'refs_xy': expected_refs_xy,
            'file_path': expected_file_path
        }
        mock_post.return_value = mock_response

        image = get_test_image()
        response = self.client.post(reverse('process_image'), {'image': image})
        self.assertEqual(response.status_code, 200)

        self.assertEqual(ObjectDetection.objects.count(), 1)
        obj = ObjectDetection.objects.latest('id')

        self.assertJSONEqual(obj.players_detections, expected_detections)
        self.assertJSONEqual(obj.players_xy, expected_players_xy)
        self.assertJSONEqual(obj.ball_xy, json.dumps(expected_ball_xy))
        self.assertJSONEqual(obj.refs_xy, json.dumps(expected_refs_xy))
        self.assertJSONEqual(obj.file_path.name, json.dumps(expected_file_path))

    @patch('requests.post')
    def test_process_image_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500

        image = get_test_image()

        self.assertEqual(ObjectDetection.objects.count(), 0)

        response = self.client.post(reverse('process_image'), {'image': image})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(ObjectDetection.objects.count(), 0)

    def test_process_image_no_image(self):
        self.assertEqual(ObjectDetection.objects.count(), 0)

        response = self.client.post(reverse('process_image'), {})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ObjectDetection.objects.count(), 0)

class LogsViewTestCase(TestCase):
    def setUp(self):
        self.url = reverse('logs')

        base_time = make_aware(datetime(2025, 4, 22))

        detection = ObjectDetection.objects.create()
        self.user = User.objects.create_user(username='logs_tester', password='password')
        self.client.login(username='logs_tester', password='password')

        self.offside1 = OffsideDecision.objects.create(
            time_uploaded=base_time,
            algorithm_decision=True,
            final_decision=True,
            detection_id=detection,
            referee_id=self.user
        )
        self.offside2 = OffsideDecision.objects.create(
            time_uploaded=base_time - timedelta(days=20),
            algorithm_decision=True,
            final_decision=True,
            detection_id=detection,
            referee_id=self.user
        )
        self.offside3 = OffsideDecision.objects.create(
            time_uploaded=base_time - timedelta(days=45),
            algorithm_decision=False,
            final_decision=False,
            detection_id=detection,
            referee_id=self.user,
        )
        self.offside4 = OffsideDecision.objects.create(
            time_uploaded=base_time - timedelta(days=400),
            algorithm_decision=True,
            final_decision=False,
            detection_id=detection,
            referee_id=self.user,
        )

    def test_logs_view_failure(self):
        self.client.logout()

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 302)

    def test_logs_view_last_week(self):
        """Test filtering offside decisions for the last week."""
        response = self.client.get(self.url, {'preset': 'last_week'})
        self.assertEqual(response.status_code, 200)

        # Check that only the offside decisions from the last week are returned
        offside_decisions = response.context['offside_decisions']
        self.assertIn(self.offside1, offside_decisions)
        self.assertNotIn(self.offside2, offside_decisions)
        self.assertNotIn(self.offside3, offside_decisions)
        self.assertNotIn(self.offside4, offside_decisions)

    def test_logs_view_last_month(self):
        """Test filtering offside decisions for the last month."""
        response = self.client.get(self.url, {'preset': 'last_month'})
        self.assertEqual(response.status_code, 200)

        # Check that only the offside decisions from the last 30 days are returned
        offside_decisions = response.context['offside_decisions']
        self.assertIn(self.offside1, offside_decisions)
        self.assertIn(self.offside2, offside_decisions)
        self.assertNotIn(self.offside3, offside_decisions)
        self.assertNotIn(self.offside4, offside_decisions)

    def test_logs_view_last_year(self):
        """Test filtering offside decisions for the last year."""
        response = self.client.get(self.url, {'preset': 'last_year'})
        self.assertEqual(response.status_code, 200)

        # Check that only the offside decisions from the last 365 days are returned
        offside_decisions = response.context['offside_decisions']

        self.assertIn(self.offside1, offside_decisions)
        self.assertIn(self.offside2, offside_decisions)
        self.assertIn(self.offside3, offside_decisions)
        self.assertNotIn(self.offside4, offside_decisions)


    def test_logs_view_start_date(self):
        response = self.client.get(reverse("logs"), {"start_date": "2025-04-01"})
        self.assertEqual(response.status_code, 200)

        decisions = response.context["offside_decisions"]
        self.assertTrue(decisions.filter(pk=self.offside1.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside2.pk).exists())
        self.assertFalse(decisions.filter(pk=self.offside3.pk).exists())
        self.assertFalse(decisions.filter(pk=self.offside4.pk).exists())

    def test_logs_view_end_date(self):
        response = self.client.get(reverse("logs"), {"end_date": "2025-03-10"})
        self.assertEqual(response.status_code, 200)

        decisions = response.context["offside_decisions"]
        self.assertFalse(decisions.filter(pk=self.offside1.pk).exists())
        self.assertFalse(decisions.filter(pk=self.offside2.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside3.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside4.pk).exists())

    def test_logs_view_start_and_end_date(self):
        response = self.client.get(reverse("logs"), {
            "start_date": "2025-03-01",
            "end_date": "2025-04-30"
        })
        self.assertEqual(response.status_code, 200)

        decisions = response.context["offside_decisions"]
        self.assertTrue(decisions.filter(pk=self.offside1.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside2.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside3.pk).exists())
        self.assertFalse(decisions.filter(pk=self.offside4.pk).exists())

    def test_logs_view_invalid_dates(self):
        response = self.client.get(reverse("logs"), {
            "start_date": "not a date",
            "end_date": "not a date"
        })

        self.assertEqual(response.status_code, 200)

        decisions = response.context["offside_decisions"]
        self.assertTrue(decisions.filter(pk=self.offside1.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside2.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside3.pk).exists())
        self.assertTrue(decisions.filter(pk=self.offside4.pk).exists())

class UpdateDetectionsViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='update_detections_tester', password='password')
        self.client.login(username='update_detections_tester', password='password')

        self.session = self.client.session

        self.session['object_detection_id'] = ObjectDetection.objects.create().id
        self.session.save()
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_update_detections_success(self):
        response = self.client.post(
            reverse('update_detections'),
            {
                'players_detections': "Some new detections",
                'players_xy': "Some new XY data"
            },
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

    def test_update_detections_failure(self):
        self.session['object_detection_id'] = "Not a detection ID"

        response = self.client.post(reverse('update_detections'), {
            'players_detections': "Some new detections",
            'players_xy': "Some new XY data"
        })

        self.assertEqual(response.status_code, 500)
