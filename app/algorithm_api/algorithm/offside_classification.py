import numpy as np
import supervision as sv

class OffsideClassification():
    def __init__(self, players_detections):
        self.GOALKEEPER_ID = 1

        self.players_detections = players_detections

        attackers_dict, defenders_dict = self.__assign_roles()

        self.attackers = sv.Detections(
            xyxy=attackers_dict['xyxy'],
            confidence=attackers_dict['confidence'],
            class_id=attackers_dict['class_id'],
            tracker_id=attackers_dict['tracker_id']
        )

        self.defenders = sv.Detections(
            xyxy=defenders_dict['xyxy'],
            confidence=defenders_dict['confidence'],
            class_id=defenders_dict['class_id'],
            tracker_id=defenders_dict['tracker_id'],
        )

        self.offside_objects = {}

    def __assign_roles(self):
        class_ids = self.players_detections['class_id']
        class_names = self.players_detections['class_name']

        team_0_indices = np.where(class_ids == 0)[0]
        team_1_indices = np.where(class_ids == 1)[0]
        goalkeeper_indices = np.where(np.char.equal(class_names, 'goalkeeper'))[0]

        if goalkeeper_indices.size == 0:
            return ValueError("No goalkeeper detected, fallback behavior not implemented.")

        # For now, assume the first goalkeeper detected is what we are interested in.
        goalkeeper_index = goalkeeper_indices[0]
        goalkeeper_team = class_ids[goalkeeper_index]

        defending_team = goalkeeper_team

        attacking_indices = team_1_indices if defending_team == 0 else team_0_indices
        defending_indices = team_0_indices if defending_team == 0 else team_1_indices

        attackers = self.__get_team_detections(attacking_indices)
        defenders = self.__get_team_detections(defending_indices)

        return attackers, defenders

    # TODO: add to database diagram
    def __get_team_detections(self, indices):
        return {
            'xyxy': self.players_detections['xyxy'][indices],
            'confidence': self.players_detections['confidence'][indices],
            'class_id': self.players_detections['class_id'][indices],
            'tracker_id' : self.players_detections['tracker_id'][indices],
            'class_name': self.players_detections['class_name'][indices],
        }

    def __get_second_defender(self, team_xy):
        highest = [-1, -float('inf')]
        second_highest = [-2,-float('inf')]

        for i, coords in enumerate(team_xy):
            x_value = coords[0]
            
            if x_value > highest[1]:
                second_highest = highest
                highest = [i, x_value]
            elif x_value > second_highest[1] and x_value != highest[1]:
                second_highest = [i, x_value]

        return second_highest[0], team_xy[second_highest[0]]

    def __setup_offside_status(self):
        for idx in range(len(self.attackers)):
            self.offside_objects[idx] = {'offside': False}

    def classify(self):
        attacking_xy = self.attackers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        defending_xy = self.defenders.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        second_defender_index, second_defender_xy = self.__get_second_defender(defending_xy)
        second_defender_id = self.defenders.tracker_id[second_defender_index]

        self.__setup_offside_status()

        player_count = 0

        for player_pos in attacking_xy:
            if player_pos[0] > second_defender_xy[0]:
                self.offside_objects[player_count]['offside'] = True
            self.offside_objects[player_count]['tracker_id'] = self.attackers.tracker_id[player_count]
            player_count += 1

        return self.offside_objects, second_defender_id
