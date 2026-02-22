import sys
import os
import requests

from dotenv import load_dotenv


sys.path.append('../../05_src/')
load_dotenv('../../05_src/.secrets')

url='https://dataservice.accuweather.com//locations/v1/cities/search?q=toronto'

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('WEATHER_API_KEY')}"
}

# get location key first for testing
response = requests.get(url, headers=headers)
locationKey = response.json()



key = locationKey[0]["Key"]
response2 = requests.get(f'https://dataservice.accuweather.com/currentconditions/v1/{key}', headers=headers)
print(response2.json())