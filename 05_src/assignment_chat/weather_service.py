import os
import sys
import requests

from openai import OpenAI
from dotenv import load_dotenv

sys.path.append('../../05_src/')
load_dotenv('../../05_src/.secrets')

# accuweather setting
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_API_URL = 'https://dataservice.accuweather.com/'
WEATHER_API_HEADER = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {WEATHER_API_KEY}"
}

# OpenAI client setting 
client = OpenAI(
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

def get_location_key(city):
    """
    Get valid location key from AccuWeather API for a single city

    Args:
        city: city name to search for
        
    Returns:
        tuple: (location_key, city_name) if valid key found, else (None, None)
    """
    try:
        print(f"Searching for: {city}")
        response = requests.get(
            f'{WEATHER_API_URL}/locations/v1/cities/search?q={city}',
            headers=WEATHER_API_HEADER
        )
        
        # Check if response is valid
        if response.status_code == 200:
            locationJson = response.json()
            
            # Check if results exist
            if locationJson and len(locationJson) > 0:
                locationKey = locationJson[0]["Key"]
                city_name = locationJson[0]["LocalizedName"]
                
                print(f"Valid location key found for {city_name}: {locationKey}")
                return locationKey, city_name
            else:
                print(f"No results found for: {city}")
        else:
            print(f"API error for {city}: {response.status_code}")
            
    except Exception as e:
        print(f"Error searching for {city}: {str(e)}")
    
    return None, None

def get_city_weather(location_key, city_name):
    """
    Get weather data and transform it to natural language using OpenAI
    
    Args:
        location_key: AccuWeather location key
        city_name: city name to display
        
    Returns:
        str: Natural language weather description
    """
    print(f"Fetching weather for {city_name}...")
    response = requests.get(
        f'{WEATHER_API_URL}/currentconditions/v1/{location_key}',
        headers=WEATHER_API_HEADER
    )

    city_weather = response.json()
    
    # Convert raw weather data to natural language using OpenAI
    weather_data = str(city_weather)
    
    user_message = f"""Please analyze this weather data from AccuWeather for {city_name} and provide a natural, conversational summary of the current weather conditions. Make it sound like you're describing the weather to a friend:

    Weather Data: {weather_data}

    Provide a brief, friendly description of the weather."""
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a friendly weather analyst assistant. Transform structured weather data into natural, conversational descriptions."},
            {"role": "user", "content": user_message}
        ],
    )
    
    return completion.choices[0].message.content


def get_weather_description(city):
    """
    Main service function to get weather description for a city
    
    Args:
        city: city name to get weather for
        
    Returns:
        str: Natural language weather description, or None if city not found
    """
    location_key, city_name = get_location_key(city)
    
    if location_key is None:
        return None
    
    weather_description = get_city_weather(location_key, city_name)
    return weather_description


# test usage
if __name__ == "__main__":
    location_key = None
    found_city = None
    
    # looping until valid location key is found
    city_name = input("Enter city name you want to know weather: ")
    location_key, found_city = get_location_key(city_name)
    
    if location_key is None:
        print("Errors to find a city key.\n")
    
    # Get and transform weather data
    weather_description = get_city_weather(location_key, found_city)
    print(f"\nWeather Summary for {found_city}:\n")
    print(weather_description)


    
    

    
