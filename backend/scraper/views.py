from django.shortcuts import render
from django.http import HttpResponse
import requests
from bs4 import BeautifulSoup

# Create your views here.
# In Ihrer views.py

def scrape_bundesliga(request):
    url = 'https://www.bundesliga.com/de/bundesliga/spieltag'
    # Beispiel: Daten von einer Webseite abrufen und verarbeiten
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        spieltag = soup.find_all('span', class_='mdc-list-item__primary-text')
        print(soup)
        return HttpResponse("Scraping erfolgreich durchgeführt")
    else:
        return HttpResponse("Scraping war nicht erfolgreich")