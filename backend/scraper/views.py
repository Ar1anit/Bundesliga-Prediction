from django.shortcuts import render
from django.http import HttpResponse
import requests
# Create your views here.
# In Ihrer views.py

def scrape_bundesliga(request):
    url = 'https://www.bundesliga.com/de/bundesliga/spieltag'
    # Beispiel: Daten von einer Webseite abrufen und verarbeiten

    return HttpResponse("Scraping erfolgreich durchgeführt")
