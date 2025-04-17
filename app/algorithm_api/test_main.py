from fastapi.testclient import TestClient

import sys
sys.path.append('/Users/jackdouet/Development/auto-off/automated-offside/app/algorithm_api')

from fastapi.testclient import TestClient
import json
from io import BytesIO
import os

from .main import app

client = TestClient(app)

# Helper function to load an image for testing
def load_test_image(image_name: str = "221_jpg.rf.a5b76a00596073c23f1254a62e945536.jpg"):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(test_dir, "test_images", image_name)
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

    # Check for expected keys in the response
    assert "ball_xy" in response_json
    assert "players_xy" in response_json
    assert "refs_xy" in response_json
    assert "players_detections" in response_json
    assert "file_path" in response_json

    # Ensure specific structure of the returned data
    assert isinstance(response_json["ball_xy"]["tracker_id"], list)
    assert isinstance(response_json["ball_xy"]["xy"], list)

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

    response = client.post(
        "/offside-classification/",
        data=json.dumps(data)
    )
    
    assert response.status_code == 200
    response_json = response.json()

    # Check for expected fields in the response
    assert "offside_status" in response_json
    assert "second_defender" in response_json
    assert "tracker_id" in response_json["second_defender"]

# Test image decoding failure (invalid image format)
def test_invalid_image():
    invalid_image = BytesIO(b"invalid_image_data")
    response = client.post(
        "/object-detection/",
        files={"image": ("invalid_image.jpg", invalid_image, "image/jpeg")},
        data={"confidence": 0.5}
    )
    
    assert response.status_code == 500
    assert "error" in response.json()

# Test offside classification with missing data
def test_offside_classification_missing_data():
    data = {}

    response = client.post(
        "/offside-classification/",
        data=json.dumps(data)
    )

    assert response.status_code == 500
    assert "error" in response.json()
