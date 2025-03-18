from cv2 import imread
from classification_helper import ClassificationHelper
from key_point_detection import KeyPointDetection
from object_detection import ObjectDetection
from visualization_helper import VisualizationHelper
import supervision as sv

# TODO: fix imports
# TODO: test algorithm

# Load in image
# TODO: move all test images into folder
# TODO: fix folder structure

# TODO: change this
test_dir = "/test_images/"
# TODO: logic for selecting images
image_route = f"{test_dir}221_jpg.rf.a5b76a00596073c23f1254a62e945536.jpg"
image = imread(image_route)

# TODO: make this changeable
# Pick weights directory
object_detection_weights_route = "weights"

# Setup classes
object_detection = ObjectDetection(weights_directory=object_detection_weights_route)
classification_helper = ClassificationHelper()
key_point_detection = KeyPointDetection()

# Run detection and split 
person_detections, ball_detections = object_detection.detect_all(image_directory=image_route)
goalkeepers_detections, players_detections, referees_detections = object_detection.split_detections(person_detections=person_detections)

# Resolve class IDs for all person detections
player_crops = [sv.crop_image(image, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = classification_helper.team_classifier(player_crops=player_crops)

goalkeepers_detections.class_id = classification_helper.resolve_goalkeepers_team_id(
    players=players_detections, goalkeepers=goalkeepers_detections
)

referees_detections.class_id -= 1

players_detections = sv.Detections.merge([
    players_detections, goalkeepers_detections
])

# Handle field detection, mapping and visualization
conf_filter, key_points = key_point_detection.detect(image=image)

visualization = VisualizationHelper(conf_filter=conf_filter, key_points=key_points)

ball_xy = visualization.transform_points(ball_detections)
players_xy = visualization.transform_points(players_detections)
refs_xy = visualization.transform_points(referees_detections)

radar_view = visualization.render_pitch(ball_xy=ball_xy, players_xy=players_xy, refs_xy=refs_xy, players_detections=players_detections)

# TODO: remove debug
sv.plot_image(radar_view)

# TODO: add offside logic

# TODO: add offside visualization