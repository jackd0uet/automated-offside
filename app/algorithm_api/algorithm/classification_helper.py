import cv2
import numpy as np
from sklearn.cluster import KMeans
import supervision as sv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

class ClassificationHelper():
    def __init__(self):
        self.player_features = []

        self.GREEN_MIN = (35, 40, 40)
        self.GREEN_MAX = (85, 255, 255)

        self.model = self.resnet()

    def resnet(self):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        return Model(inputs=base_model.input, outputs=base_model.output)
    
    def extract_deep_features(
            self,
            image
    ) -> np.ndarray:
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = self.model.predict(image)

        return features.flatten()

    def team_classifier(
            self,
            player_crops
    ) -> np.ndarray:

        for cropped_player in player_crops:
            cropped_player = np.array(cropped_player, dtype=np.uint8)
            cropped_player = cv2.cvtColor(cropped_player, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(cropped_player, self.GREEN_MIN, self.GREEN_MAX)
            inverted_mask = cv2.bitwise_not(mask)

            cropped_player = cv2.bitwise_and(cropped_player, cropped_player, mask=inverted_mask)
        
            deep_features = self.extract_deep_features(cropped_player)
            if not deep_features.any():
                return ValueError("No player detected")

            self.player_features.append(np.hstack([deep_features]))

        x = np.array(self.player_features, dtype=np.float32)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        
        labels = kmeans.fit_predict(x)
        
        return np.array(labels.flatten())

    def resolve_goalkeepers_team_id(
            self,
            players: sv.Detections,
            goalkeepers: sv.Detections,
            threshold: float = 30.0,
            alpha: float = 0.5
    ) -> np.ndarray:

        try:
            # Get positions of keepers and players
            goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            print("players_xy shape:", players_xy.shape)
            print("players.class_id shape:", players.class_id.shape)
            print("players.class_id:", players.class_id)

            # Split players into teams
            team_0_players = players_xy[players.class_id == 0]
            team_1_players = players_xy[players.class_id == 1]

            # Get centroids for each team
            team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
            team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)

            # For recording final results
            goalkeepers_team_id = []
            
            for goalkeeper_xy in goalkeepers_xy:
                # Centroid distances
                dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
                dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)

                # Normalised distances
                max_dist = max(dist_0, dist_1)
                norm_dist_0 = dist_0 / max_dist if max_dist > 0 else 0
                norm_dist_1 = dist_1 / max_dist if max_dist > 0 else 0

                # Nearby players
                near_team_0 = np.sum(np.linalg.norm(team_0_players - goalkeeper_xy, axis=1) < threshold)
                near_team_1 = np.sum(np.linalg.norm(team_1_players - goalkeeper_xy, axis=1) < threshold)

                # Normalise player counts
                total_near_players = near_team_0 + near_team_1
                norm_near_0 = 1 - (near_team_0 / total_near_players) if total_near_players > 0 else 0.5
                norm_near_1 = 1 - (near_team_1 / total_near_players) if total_near_players > 0 else 0.

                # Final scores
                score_0 = alpha * norm_dist_0 + (1 - alpha) * norm_near_0
                score_1 = alpha * norm_dist_1 + (1 - alpha) * norm_near_1

                # Assign keeper
                goalkeepers_team_id.append(0 if score_0 < score_1 else 1)

            return np.array(goalkeepers_team_id)
        
        except Exception as e:
            raise ValueError("Goalkeeper classification failed")
