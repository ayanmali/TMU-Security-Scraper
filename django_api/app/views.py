from django.shortcuts import get_object_or_404#, render
from django.http import HttpResponse, JsonResponse 
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from .models import Incident # importing the Postgres model
from .serializers import IncidentSerializer

DEFAULT_TO_RETRIEVE = 5

# class IncidentViewSet(viewsets.ModelViewSet):
#     queryset = Incident.objects.all().order_by('id').values()
#     serializer_class = IncidentSerializer
#     #authentication_classes = (SessionAuthentication, BasicAuthentication, TokenAuthentication)

#     # Viewset
#     @action(detail=False, methods=['get'], url_path='custom/(?P<incident_id>[^/.]+)')
#     def get_by_id(self, request, incident_id):
#         incident = get_object_or_404(Incident, id=incident_id)
#         serialized = self.get_serializer(incident)
#         return Response(serialized.data)

def num_input_validation(param, default=DEFAULT_TO_RETRIEVE, error_msg=""):
    if (param.isdigit() and int(param) < 1) or param is None or not param.isdigit():
        return Response({
            'error' : error_msg
            }, status=status.HTTP_400_BAD_REQUEST)
    
    return None

# Create your views here.
def index(request):
    return JsonResponse({"result" : "Welcome to the TMU Security Incidents Website"})

"""
Endpoint to retrieve a single incident based on its ID in the database.
"""
class IncidentDetailView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, incident_id):
        # ADD ERROR HANDLING

        item = get_object_or_404(Incident, id=incident_id)
        serializer = IncidentSerializer(item)
        return Response(serializer.data)
    
"""
Endpoint to retrieve the N most recent incidents
"""
class RecentIncidents(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Getting the query parameter that specifies how many incidents to retrieve
        limit = request.query_params.get('limit', None)
        input_val = num_input_validation(limit, error_msg="The number of incidents must be a valid number greater than or equal to 1.")
        if input_val:
            return input_val
        limit = DEFAULT_TO_RETRIEVE if limit == "" else int(limit)
        # if limit == "" or limit is None:
        #     limit = 5
        # elif (limit.isdigit() and int(limit) < 1) or not limit.isdigit():
        #     return Response({
        #         'error' : 'The number of incidents must be a valid number greater than or equal to 1.'
        #         }, status=status.HTTP_400_BAD_REQUEST)

        queryset = Incident.objects.all().order_by('-dateofincident')[:int(limit)]

        # Seralize the queried data
        serializer = IncidentSerializer(queryset, many=True)

        return Response(serializer.data)
    
    """
    Endpoint to retrieve all incidents in a paginated format.
    """
    class PaginatedIncidents(APIView):
        authentication_classes = [TokenAuthentication]
        permission_classes = [IsAuthenticated]
        
        def get(self, request):
            offset = request.query_params.get('offset', None)
            limit = request.query_params.get('limit', None)

            if offset == "" or offset is None:
                offset = 5
            elif (offset.isdigit() and int(limit) < 1) or not limit.isdigit():
                return Response({
                    'error' : 'The number of incidents must be a valid number greater than or equal to 1.'
                    }, status=status.HTTP_400_BAD_REQUEST)


