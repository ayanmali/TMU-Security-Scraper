from django.shortcuts import get_object_or_404#, render
from django.http import HttpResponse, JsonResponse 
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from .postgres_params import USER, PASSWORD, HOST, PORT, DBNAME

from .models import Incident # importing the Postgres model
from .serializers import IncidentSerializer

from openai import OpenAI

from sqlalchemy import create_engine

import numpy as np
import pandas as pd

# Importing ML algorithms
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers/TMU-ML/TMU-Security-Scraper')
from search import get_search_results, LOCDETAILS_EMBED_COLUMN_NAME, LOCDESCR_EMBED_COLUMN_NAME
from recommend_tfidf_algo import get_recommendations, load_and_transform_data, parse_incident_identifier, train_model, N_NEIGHBORS

DEFAULT_TO_RETRIEVE = 5 # Default number of incidents to retrieve if a value isn't specified in the query parameters
PER_PAGE = 20 # Number of incidents to be displayed on a given page on the website
TABLE_NAME = "incidents"

client = OpenAI()
engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}')

# Setting up the DataFrame to use for searches/retrieving recent incidents
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
# Convert string representations of vectors to numpy arrays
df[LOCDESCR_EMBED_COLUMN_NAME] = df[LOCDESCR_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
# Convert string representations of vectors to numpy arrays
df[LOCDETAILS_EMBED_COLUMN_NAME] = df[LOCDETAILS_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

# Creating the DataFrame to use for recommendations
recommend_df, _ = load_and_transform_data(df)
# Training the recommendation model
knn = train_model(recommend_df, N_NEIGHBORS)

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

def num_input_validation(params, defaults, error_msgs):
    res = []
    
    for idx, param in enumerate(params): 
        if (param is None or param == ""):
            res.append(defaults[idx])
        
        elif (param.isdigit()):
            res.append(int(param))
        
        elif (param.isdigit() and int(param) < 1) or (not param.isdigit()):
            res.append(Response({
                'error' : error_msgs[idx]
                }, status=status.HTTP_400_BAD_REQUEST))
        
    return res

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
        # Validating the query parameter
        input_val = num_input_validation([limit], defaults=[DEFAULT_TO_RETRIEVE], error_msgs=["The number of incidents must be a valid number greater than or equal to 1."])[0]
        if type(input_val) is Response:
            return input_val
        limit = input_val
        
        # Retrieving records from the DB
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

        # Validating the query parameters
        offset, limit = num_input_validation([offset, limit], defaults=[0, PER_PAGE], error_msgs=[
            "The offset must be a valid number greater than or equal to 0.",
            "The limit must be a valid number greater than or equal to 1."]
        )

        for param in [offset, limit]:
            if type(param) is Response:
                return param 
            
        # Querying the DB
        total_count = Incident.objects.count()
        start, end = offset, offset + limit

        incidents = Incident.objects.all().order_by('-dateofincident')[start:end]

        serializer = IncidentSerializer(incidents, many=True)

        total_pages = (total_count + limit - 1) // limit 

        curr_page = offset // limit 

        data = {
            'results': serializer.data,
            'pagination': {
                'total_records': total_count,
                'total_pages': total_pages,
                'current_page': curr_page,
                'limit': limit,
                'offset': offset,
                'has_next': end < total_count,
                'has_previous': offset > 0
            }
        } 

        return Response(data)
    
class SearchIncidents(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        query = request.query_params.get('query', None)
        # Validating the search query
        if query is None or query == "":
            return Response({
                'error' : "Search query cannot be empty."
                }, status=status.HTTP_400_BAD_REQUEST)
        if len(query) < 4:
            return Response({
                'error' : "Search query must contain at least 4 characters."
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validating the limit query parameter
        limit = request.query_params.get('limit', None)
        # Validating the query parameter
        input_val = num_input_validation([limit], defaults=[DEFAULT_TO_RETRIEVE], error_msgs=["The number of incidents must be a valid number greater than or equal to 1."])[0]
        if type(input_val) is Response:
            return input_val
        limit = input_val

        # Retrieving the IDs of matching records from the DB
        search_results = get_search_results(client,query,vector_column=0, df=df, n=limit)['id'].values

        # Querying the Incident Model
        incidents = Incident.objects.filter(pk__in=search_results)

        serializer = IncidentSerializer(incidents, many=True)

        return Response({'count' : limit,
                         'results' : serializer.data})

