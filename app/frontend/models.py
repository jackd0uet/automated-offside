from django.contrib.auth.models import User
from django.db import models

class DecisionChoices(models.TextChoices):
    ONSIDE = 'onside', 'Onside'
    OFFSIDE = 'offside', 'Offside'

class ObjectDetection(models.Model):
    id = models.AutoField(db_column='id', primary_key=True)
    players_detections = models.TextField(db_column='players_detections')
    players_xy = models.TextField(db_column='players_xy')
    ball_xy = models.TextField(db_column='ball_xy')
    refs_xy = models.TextField(db_column='refs_xy')
    file_path = models.FileField(db_column='file_path', null=False, blank=False)

    class Meta:
        managed = True
        db_table = 'object_detection'

    def __str__(self) -> str:
        return f"{self.id} | {self.file_path}"

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
        return f"{self.id} | {self.detection_id.id} | {self.algorithm_decision} | {self.final_decision} | {self.time_uploaded} | {self.time_decided}"
   