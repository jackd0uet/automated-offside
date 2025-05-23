import numpy as np
import supervision as sv

class OffsideClassification():
    '''
    This class takes player detection objects from object detection models and makes offside decisions.
    '''
    def __init__(self, players_detections, defending_team=None):
        # Optional, if the defending team is known, fallback if there is no goalkeeper.
        self.defending_team = defending_team

        self.players_detections = players_detections

        attackers_dict, defenders_dict = self.__assign_roles()

        # Supervision Detection objects for both teams.
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

        # Stores whether attackers are offside.
        self.offside_objects = {}

    def __assign_roles(self):
        class_ids = self.players_detections['class_id']
        class_names = self.players_detections['class_name']

        # Split detections into two teams.
        team_0_indices = np.where(class_ids == 0)[0]
        team_1_indices = np.where(class_ids == 1)[0]

        # Get goalkeeper.
        goalkeeper_indices = np.where(np.char.equal(class_names, 'goalkeeper'))[0]

        # This is actually the default behavior, defending team will already be set if there is no goalkeeper.
        if self.defending_team == None:
            goalkeeper_index = goalkeeper_indices[0]
            goalkeeper_team = class_ids[goalkeeper_index]

            self.defending_team = goalkeeper_team

        # Assign the teams to attack or defend.
        attacking_indices = team_1_indices if self.defending_team == 0 else team_0_indices
        defending_indices = team_0_indices if self.defending_team == 0 else team_1_indices

        attackers = self.__get_team_detections(attacking_indices)
        defenders = self.__get_team_detections(defending_indices)

        return attackers, defenders

    # TODO: add to class diagram
    def __get_team_detections(self, indices):
        return {
            'xyxy': self.players_detections['xyxy'][indices],
            'confidence': self.players_detections['confidence'][indices],
            'class_id': self.players_detections['class_id'][indices],
            'tracker_id' : self.players_detections['tracker_id'][indices],
            'class_name': self.players_detections['class_name'][indices],
        }

    def __get_second_defender(self, team_xy):
        # sort the team based on their position on the field.
        sorted_team = sorted(enumerate(team_xy), key=lambda x: x[1][0], reverse=True)

        # If there are at least two defenders, return the second defender.
        if len(sorted_team) >= 2:
            second_defender_index = sorted_team[1][0]
            return second_defender_index, team_xy[second_defender_index]
        else:
            return None, None

    def __setup_offside_status(self):
        # Populate offside status dict, assign all attackers onside initially.
        for idx in range(len(self.attackers)):
            self.offside_objects[idx] = {'offside': False}

    def classify(self):
        # Use Supervision Position to get the middle bottom of the bounding box.
        # This is a slightly looser interpretation of the offside rule than is used normally,
        # but as the goal of this project is speed and efficiency for lower league use this approach
        # works better.
        attacking_xy = self.attackers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        defending_xy = self.defenders.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        # Find the second defender.
        second_defender_index, second_defender_xy = self.__get_second_defender(defending_xy)
        second_defender_id = self.defenders.tracker_id[second_defender_index]

        self.__setup_offside_status()

        # For each attacking player determine if they are beyond the second last defender and are therefore offside.
        player_count = 0
        for player_pos in attacking_xy:
            if second_defender_index != None and player_pos[0] > second_defender_xy[0]:
                self.offside_objects[player_count]['offside'] = True
            self.offside_objects[player_count]['tracker_id'] = self.attackers.tracker_id[player_count]
            player_count += 1

        return self.offside_objects, second_defender_id
