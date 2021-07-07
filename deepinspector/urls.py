from django.urls import include, path
from . import views

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('rlassist/predicted', views.RlAssistPredictedView.as_view(), name="predicted"),
]
