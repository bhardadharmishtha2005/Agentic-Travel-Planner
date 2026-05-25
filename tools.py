import json
import requests
from langchain.tools import tool

# 1. Tool to search Flight Data
@tool
def search_flights(source: str, destination: str) -> str:
    """Searches for available flights between a source and a destination from flights.json.
    Input should be the source city and destination city name."""
    try:
        with open("flights.json", "r") as file:
            flights = json.load(file)
        
        matching_flights = [
            f for f in flights 
            if f.get("source", "").lower() == source.lower() 
            and f.get("destination", "").lower() == destination.lower()
        ]
        
        if not matching_flights:
            return f"No direct flights found from {source} to {destination} in our records."
        return json.dumps(matching_flights, indent=2)
    except FileNotFoundError:
        return "Error: flights.json file not found in the directory."
    except Exception as e:
        return f"Error reading flight data: {str(e)}"


# 2. Tool to fetch Live Weather using your exact required Open-Meteo API URL
@tool
def get_live_weather(latitude: float, longitude: float) -> str:
    """Fetches a real-time daily maximum temperature forecast for a location given its latitude and longitude."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max&timezone=auto"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            daily_forecast = data.get("daily", {})
            if daily_forecast:
                return json.dumps(daily_forecast, indent=2)
        return "Weather data is currently unavailable from the API."
    except Exception as e:
        return f"Error connecting to weather API: {str(e)}"


# 3. Tool to search Hotel Data
@tool
def search_hotels(city: str, max_budget: float = None) -> str:
    """Searches for hotels in a specific destination city. 
    Can optionally filter by a maximum budget limit per night."""
    try:
        with open("hotels.json", "r") as file:
            hotels = json.load(file)
            
        matching_hotels = [
            h for h in hotels 
            if h.get("city", "").lower() == city.lower()
        ]
        
        if max_budget is not None:
            matching_hotels = [h for h in matching_hotels if h.get("price_per_night", 0) <= float(max_budget)]
            
        if not matching_hotels:
            return f"No hotels found in {city} within the specified budget parameters."
        return json.dumps(matching_hotels, indent=2)
    except FileNotFoundError:
        return "Error: hotels.json file not found in the directory."
    except Exception as e:
        return f"Error reading hotel data: {str(e)}"


# 4. Tool to search Attractions/Places Data
@tool
def search_places(city: str) -> str:
    """Searches for popular tourist attractions, coordinates (latitude/longitude), and ratings for a destination city."""
    try:
        with open("places.json", "r") as file:
            places = json.load(file)
            
        matching_places = [
            p for p in places 
            if p.get("city", "").lower() == city.lower()
        ]
        
        if not matching_places:
            return f"No local attractions data found for {city}."
        return json.dumps(matching_places, indent=2)
    except FileNotFoundError:
        return "Error: places.json file not found in the directory."
    except Exception as e:
        return f"Error reading places data: {str(e)}"