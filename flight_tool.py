
# Example free API usage
# - AviationStack


# create api key
# https://aviationstack.com/ 
# pip install requests


    
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

def search_flights(dep_iata: str, arr_iata: str):
    """
    Fetches live flight information filtered by both departure and arrival airport codes.
    """
    url = "http://api.aviationstack.com/v1/flights"

    # 1. Update params to map both departure and arrival airport codes
    params = {
        "access_key": API_KEY,
        "limit": 5,
        "dep_iata": dep_iata.strip().upper(),
        "arr_iata": arr_iata.strip().upper()
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
    except Exception as e:
        return f"Error connecting to flight API: {str(e)}"

    flights = []

    # 2. Fixed the "if data in data" logic bug and verified it's a list
    if "data" in data and isinstance(data["data"], list):
        for flight in data["data"][:5]:
            airline = flight.get("airline", {}).get("name", "Unknown")
            departure = flight.get("departure", {}).get("airport", "Unknown")
            arrival = flight.get("arrival", {}).get("airport", "Unknown")
            status = flight.get("flight_status", "Unknown")

            flights.append(
                f"Airline: {airline}\nDeparture: {departure}\nArrival: {arrival}\nStatus: {status}\n"
            )

    # 3. Graceful fallback text if the API returns an empty flight array
    if not flights:
        return f"No live scheduled flights found right now for the route: {dep_iata} ➔ {arr_iata}."

    return "\n".join(flights)