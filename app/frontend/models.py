from django.db import models

# TODO: update database diagram
class ObjectDetection(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    players_detections = models.CharField(db_column="players_detections", max_length=2048)
    players_xy = models.CharField(db_column="players_xy", max_length=2048)
    ball_xy = models.CharField(db_column="ball_xy", max_length=2048)
    refs_xy = models.CharField(db_column="refs_xy", max_length=2048)
    file_path = models.CharField(db_column="file_path", null=False, blank=False, max_length=2048)

    class Meta:
        managed = True
        db_table = "object_detection"

    def __str__(self) -> str:
        return f"{self.id} | {self.player_detections} | {self.players_xy} | {self.ball_xy} | {self.ref_xy} | {self.file_path}"

class OffsideDecision(models.Model):
    id = models.AutoField(db_column='id', primary_key=True),
    detection_id = models.ForeignKey(ObjectDetection, models.PROTECT, db_column='detection_id', blank=False, null=False),
    algorithm_offside = models.BooleanField(db_column='algorithm_decision', null=False, blank=False)
    final_offside = models.BooleanField(db_column="final_decision", null=True, blank=False)
    time_uploaded = models.DateField(db_column="time_uploaded", null=False, blank=False)
    time_decided = models.DateField(db_column="time_decided", null=True, blank=False)

    class Meta:
        managed = True
        db_table = "offside_decision"

    def __str__(self) -> str:
        return f"{self.id} | {self.detection_id}  |{self.algorithm_offside} | {self.final_offside} | {self.time_uploaded} | {self.time_decided}"
   