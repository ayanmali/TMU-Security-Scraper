from django.db import models
from pgvector.django import VectorField

from .postgres_params import TABLE_NAME

# Create your models here.
class Incident(models.Model):
    id = models.IntegerField(primary_key=True)
    incidenttype = models.TextField()
    location = models.TextField()
    page = models.TextField()
    incidentdetails = models.TextField()
    description = models.TextField()
    dateofincident = models.DateTimeField()
    datereported = models.DateTimeField()
    dateposted = models.DateTimeField()

    # Vector embedding columns
    locationembed = VectorField(dimensions=128)
    locdescrembed = VectorField(dimensions=384)
    descrembed = VectorField(dimensions=256)
    detailsembed = VectorField(dimensions=256)
    locdetailsembed = VectorField(dimensions=384)

    def __str__(self):
        return self.incidenttype + " - " + self.location

    # Table metadata
    class Meta:
        db_table = TABLE_NAME
        managed = False # Tells Django not to manage this table's schema