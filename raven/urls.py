from django.urls import include, path
from . import views

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('sr2ml', views.SR2MLView.as_view(), name="sr2ml"),
]
