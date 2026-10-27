import os
import sys

import requests

API = "https://api.openweathermap.org/data/2.5/weather"


def current(city: str) -> dict:
    key = os.environ["OPENWEATHER_API_KEY"]
    resp = requests.get(API, params={"q": city, "appid": key, "units": "metric"}, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    data = current(sys.argv[1])
    print(f"{data['main']['temp']} C, {data['weather'][0]['description']}")
