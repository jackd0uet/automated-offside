from datetime import datetime
from fastapi.testclient import TestClient
from io import BytesIO
import json
import logging
import os
from unittest import mock

from main import app

client = TestClient(app)

# Helper function to load an image for testing
def load_test_image(image_name: str = "221_jpg.rf.a5b76a00596073c23f1254a62e945536.jpg"):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(test_dir, "images", image_name)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return BytesIO(image_data)

# Test for the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Algorithm API is running"}

# Test for object detection endpoint
def test_object_detection():
    image = load_test_image()
    response = client.post(
        "/object-detection/",
        files={"image": ("test_image.jpg", image, "image/jpeg")},
        data={"confidence": 0.6}
    )
    
    assert response.status_code == 200
    response_json = response.json()

    # Basic structure validation
    expected_keys = [
        "ball_xy", "players_xy", "refs_xy", "players_detections", "file_path"
    ]
    for key in expected_keys:
        assert key in response_json

    # ball_xy structure
    ball_xy = response_json["ball_xy"]
    assert isinstance(ball_xy, dict)
    assert isinstance(ball_xy.get("tracker_id"), list)
    assert isinstance(ball_xy.get("xy"), list)
    assert all(isinstance(tid, int) for tid in ball_xy["tracker_id"])
    for xy in ball_xy["xy"]:
        assert isinstance(xy, list) and len(xy) == 2
        assert all(isinstance(coord, (int, float)) for coord in xy)

    # players_xy structure
    players_xy = response_json["players_xy"]
    assert isinstance(players_xy, dict)
    assert isinstance(players_xy.get("tracker_id"), list)
    assert isinstance(players_xy.get("xy"), list)
    assert all(isinstance(tid, int) for tid in players_xy["tracker_id"])
    for xy in players_xy["xy"]:
        assert isinstance(xy, list) and len(xy) == 2
        assert all(isinstance(coord, (int, float)) for coord in xy)

    # refs_xy structure
    refs_xy = response_json["refs_xy"]
    assert isinstance(refs_xy, dict)
    assert isinstance(refs_xy.get("tracker_id"), list)
    assert isinstance(refs_xy.get("xy"), list)
    assert all(isinstance(tid, int) for tid in refs_xy["tracker_id"])
    for xy in refs_xy["xy"]:
        assert isinstance(xy, list) and len(xy) == 2
        assert all(isinstance(coord, (int, float)) for coord in xy)

    # players_detections structure
    players_detections = response_json["players_detections"]
    assert isinstance(players_detections, dict)
    for key in ["xyxy", "confidence", "class_id", "tracker_id", "class_name"]:
        assert key in players_detections

    # Validate players_detections["xyxy"]
    xyxy = players_detections["xyxy"]
    assert isinstance(xyxy, list)
    for box in xyxy:
        assert isinstance(box, list) and len(box) == 4
        assert all(isinstance(coord, (int, float)) for coord in box)

    # Validate players_detections["confidence"]
    confidence = players_detections["confidence"]
    assert isinstance(confidence, list)
    assert all(isinstance(conf, float) for conf in confidence)

    # Validate players_detections["class_id"]
    class_id = players_detections["class_id"]
    assert isinstance(class_id, list)
    assert all(isinstance(cid, int) for cid in class_id)

    # Validate players_detections["tracker_id"]
    tracker_id = players_detections["tracker_id"]
    assert isinstance(tracker_id, list)
    assert all(isinstance(tid, int) for tid in tracker_id)

    # Validate players_detections["class_name"]
    class_name = players_detections["class_name"]
    assert isinstance(class_name, list)
    assert all(isinstance(cname, str) for cname in class_name)
    assert "goalkeeper" in class_name

    # Check file_path is a string
    assert isinstance(response_json["file_path"], str)

# Test object detection 400 on string confidence input
def test_object_detection_bad_confidence():
    image = load_test_image()
    response = client.post(
        "/object-detection/",
        files={"image": ("test_image.jpg", image, "image/jpeg")},
        data={"confidence": 'this is a bad value'}
    )

    assert response.status_code == 400

# Test object detection 400 on no image
def test_object_detection_no_image():
    response = client.post(
        "/object-detection/",
        data={"confidence": 0.6}
    )

    assert response.status_code == 400

# Test image saving failure (shutil error)
def test_object_detection_image_saving_failure(caplog):
    with mock.patch("shutil.copyfileobj", side_effect=IOError("Simulated copy failure")):
        with caplog.at_level(logging.WARNING):
            image = load_test_image()
            client.post(
                "/object-detection/",
                files={"image": ("test_image.jpg", image, "image/jpeg")},
                data={"confidence": 0.6}
            )

        assert any("Image saving failed" in message for message in caplog.messages)

# Test image decoding failure (invalid image format)
def test_object_detection_invalid_image():
    invalid_image = BytesIO(b"invalid_image_data")
    response = client.post(
        "/object-detection/",
        files={"image": ("invalid_image.jpg", invalid_image, "image/jpeg")},
        data={"confidence": 0.5}
    )

    assert response.status_code == 500
    assert "error" in response.json()

# Test for offside classification endpoint
def test_offside_classification():
    # Sample input data for the offside classification
    data = {
        "detection_data": {
            "players_detections": {
                "xyxy": [
                    [383.3307800292969, 536.4382934570312, 448.2730712890625, 651.3054809570312],
                    [976.3394165039062, 452.2207946777344, 1034.7899169921875, 544.2407836914062],
                    [534.8789672851562, 314.8357849121094, 584.1883544921875, 406.7942810058594],
                    [1169.178466796875, 510.8219909667969, 1220.56787109375, 621.6669921875],
                    [790.4156494140625, 585.863037109375, 833.4833374023438, 695.3049926757812],
                    [706.2911376953125, 332.18145751953125, 742.2997436523438, 412.7750549316406],
                    [1370.7593994140625, 538.3462524414062, 1428.5467529296875, 637.0547485351562],
                    [1301.8985595703125, 526.4086303710938, 1347.43505859375, 622.3670654296875],
                    [671.8126831054688, 746.4970092773438, 735.9869384765625, 869.0905151367188],
                    [1481.7154541015625, 307.19879150390625, 1533.6932373046875, 385.92901611328125]
                ],
                "confidence": [
                    0.8916431069374084, 0.889184296131134, 0.8841703534126282, 0.8703374266624451,
                    0.8560696840286255, 0.8539551496505737, 0.8536351919174194, 0.8325600624084473,
                    0.8115831613540649, 0.8662241101264954
                ],
                "class_id": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                "tracker_id": [0, 1, 2, 3, 5, 6, 7, 9, 10, 4],
                "class_name": [
                    "player", "player", "player", "player", "player", "player", "player", "player", "player", "goalkeeper"
                ]
            },
        },
        "defending_team": None,
    }
    classification_start = datetime.now()

    response = client.post(
        "/offside-classification/",
        data=json.dumps(data)
    )

    classification_end = datetime.now()

    print(f"Offside classification took {classification_end - classification_start} seconds to run")
    
    assert response.status_code == 200
    response_json = response.json()

    # Check for expected fields in the response
    assert "offside_status" in response_json
    assert "second_defender" in response_json
    assert "tracker_id" in response_json["second_defender"]

# Test offside classification with missing data
def test_offside_classification_missing_data():
    data = {}

    response = client.post(
        "/offside-classification/",
        data=json.dumps(data)
    )

    assert response.status_code == 500
    assert "error" in response.json()
