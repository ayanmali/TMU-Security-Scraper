from django.urls import path#, include
#from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

from . import views

# router = DefaultRouter()
# router.register(r'items', views.IncidentViewSet)

urlpatterns = [
    path('', views.index, name='index'),
    #path('getone/<int:incident_id>/', views.get_incident_by_id, name='get_incident_by_id'),
    #path('test/', include(router.urls))
    path('getone/<int:incident_id>', views.IncidentDetailView.as_view(), name='get-one'),
    path('getrecent/', views.RecentIncidents.as_view(), name='get-recent'),
    #path('getincidents/', views.PaginatedIncidents.as_view(), name='paginated-incidents'),

    # Endpoint for providing a user with an auth token
    path('api-token-auth/', obtain_auth_token, name='api-token'),
]

