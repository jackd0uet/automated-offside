from django.db import models

# Create your models here.

class OffsideDecision(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    file_name = models.CharField(db_column="filename", null=False, blank=False, max_length=2048)
    algorithm_offside = models.BooleanField(db_column='algorithm_offside', null=False, blank=False)
    final_offside = models.BooleanField(db_column="final_offside", null=True, blank=False)
    time_uploaded = models.DateField(db_column="time_uploaded", null=False, blank=False)
    time_decided = models.DateField(db_column="time_decided", null=True, blank=False)

    class Meta:
        managed = True
        db_table = "offside_decision"

    def __str__(self) -> str:
        return f"{self.id} | {self.file_name} |{self.algorithm_offside} | {self.final_offside} | {self.time_uploaded} | {self.time_decided}"
    
class ObjectDetection(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    decision_id = models.ForeignKey(OffsideDecision, models.PROTECT, db_column='decision_id', blank=False, null=False)
    player_detections = models.CharField(db_column="player_detections", max_length=2048)
    keeper_detections = models.CharField(db_column="keeper_detections", max_length=2048)
    ball_detections = models.CharField(db_column="ball_detections", max_length=2048)
    ref_detections = models.CharField(db_column="ref_detections", max_length=2048)

    class Meta:
        managed = True
        db_table = "object_detection"

    def __str__(self) -> str:
        return f"{self.id} | {self.decision_id} | {self.player_detections} | {self.keeper_detections} | {self.ball_detections} | {self.ref_detections}"
