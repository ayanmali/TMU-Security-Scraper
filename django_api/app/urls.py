from django.urls import path#, include
#from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

# Importing the views defined in views.py
from . import views

# router = DefaultRouter()
# router.register(r'items', views.IncidentViewSet)

# Defining API endpoints and the views that handle each one
urlpatterns = [
    path('', views.index, name='index'),
    #path('getone/<int:incident_id>/', views.get_incident_by_id, name='get_incident_by_id'),
    #path('test/', include(router.urls))
    path('getone/<int:incident_id>', views.IncidentDetailView.as_view(), name='get-one'),
    path('getrecent/', views.RecentIncidents.as_view(), name='get-recent'),
    path('getincidents/', views.PaginatedIncidents.as_view(), name='paginated-incidents'),
    path('search/', views.SearchIncidents.as_view(), name='search-incidents'),
    path('recommend/<str:date_ident>', views.RecommendIncidents.as_view(), name='recommend-incidents'),

    # Endpoint for providing a user with an auth token
    path('api-token-auth/', obtain_auth_token, name='api-token'),
]

