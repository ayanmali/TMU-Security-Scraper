from rest_framework import serializers
from .models import Incident

class IncidentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incident
        fields = ['id', 'incidenttype', 'location', 'page', 'incidentdetails', 'description', 'dateofincident', 'datereported', 'dateposted']
        #fields = '__all__' # specifies which fields are included in the serialized Incident record
    