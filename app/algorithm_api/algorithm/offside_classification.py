import supervision as sv

class OffsideClassification():
    def __init__(self, players_detections, all_detections):
        self.GOALKEEPER_ID = 1

        self.all_detections = all_detections
        self.players_detections = players_detections

        self.attackers, self.defenders = self.__assign_roles()

        self.offside_status = {}

    def __assign_roles(self):
        team_0 = self.players_detections[self.players_detections.class_id == 0]
        team_1 = self.players_detections[self.players_detections.class_id == 1]

        if self.__detect_goalkeeper():
            defending_team = 0 if len(team_0[team_0.data['class_name'] == 'goalkeeper']) > 0 else 1
        else:
            # TODO: Work out fallback behavior if no keeper
            return ValueError("No goalkeeper detected, fallback behavior not implemented.")

        return (team_1, team_0) if defending_team == 0 else (team_0, team_1)

    def __detect_goalkeeper(self):
        if len(self.all_detections[self.all_detections.class_id == self.GOALKEEPER_ID]) > 0:
            return True

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

        return team_xy[second_highest[0]]

    def __setup_offside_status(self):
        for idx in enumerate(self.attackers):
            self.offside_status[idx] = {"offside": False}

    def classify(self):
        attacking_xy = self.attackers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        defending_xy = self.defenders.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        second_defender_xy = self.__get_second_defender(defending_xy)

        self.__setup_offside_status()

        player_count = 0

        for player_pos in attacking_xy:
            if player_pos[0] > second_defender_xy[0]:
                self.offside_status[player_count]['offside'] = True
            player_count += 1

        return self.offside_status
