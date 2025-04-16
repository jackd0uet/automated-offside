from django.contrib.auth.models import User
from django.db import models

# TODO: update database diagram

class DecisionChoices(models.TextChoices):
    ONSIDE = 'onside', 'Onside'
    OFFSIDE = 'offside', 'Offside'

class ObjectDetection(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    players_detections = models.CharField(db_column='players_detections', max_length=4096)
    players_xy = models.CharField(db_column='players_xy', max_length=4096)
    ball_xy = models.CharField(db_column='ball_xy', max_length=4096)
    refs_xy = models.CharField(db_column='refs_xy', max_length=4096)
    file_path = models.CharField(db_column='file_path', null=False, blank=False, max_length=4096)

    class Meta:
        managed = True
        db_table = 'object_detection'

    def __str__(self) -> str:
        return f"{self.id} | {self.players_detections} | {self.players_xy} | {self.ball_xy} | {self.refs_xy} | {self.file_path}"

class OffsideDecision(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    detection_id = models.ForeignKey(ObjectDetection, models.PROTECT, db_column='detection_id', blank=False, null=False)
    referee_id = models.ForeignKey(User, models.PROTECT, db_column='referee_id', blank=False, null=False)
    algorithm_decision = models.CharField(db_column='algorithm_decision', choices=DecisionChoices.choices, null=False, blank=False, max_length=10)
    final_decision = models.CharField(db_column='final_decision', choices=DecisionChoices.choices, null=True, blank=False, max_length=10)
    time_uploaded = models.DateTimeField(db_column='time_uploaded', null=False, blank=False)
    time_decided = models.DateTimeField(db_column='time_decided', null=True, blank=False)

    class Meta:
        managed = True
        db_table = 'offside_decision'

    def __str__(self) -> str:
        return f"{self.id} | {self.detection_id}  |{self.algorithm_decision} | {self.final_decision} | {self.time_uploaded} | {self.time_decided}"
   