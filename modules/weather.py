"""
Weather lookup via Open-Meteo (no API key required).
Detects weather-related queries and returns a plain-English summary
that is injected into the LLM context.
"""

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

_WMO: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

_WEATHER_KW = re.compile(
    r"\b(weather|temperature|forecast|raining|snowing|sunny|cloudy|hot|cold|warm|humid|umbrella)\b",
    re.I,
)
# Matches "weather in Paris" / "weather for London" / "in Tokyo"
_CITY_RE = re.compile(
    r"\b(?:weather\b.*?\b(?:in|for)|(?:in|for))\s+([A-Za-z][A-Za-z\s]{1,28}?)(?:[?.,!]|\s*$)",
    re.I,
)


def is_weather_query(text: str) -> bool:
    return bool(_WEATHER_KW.search(text))


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=6) as r:
        return json.loads(r.read())


def _geocode(city: str) -> Optional[tuple[float, float, str]]:
    enc = urllib.parse.quote(city)
    data = _fetch(
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={enc}&count=1&language=en&format=json"
    )
    results = data.get("results")
    if not results:
        return None
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city)


def get_weather_summary(
    text: str,
    default_lat: float,
    default_lon: float,
    default_city: str,
) -> Optional[str]:
    """
    Returns a one-line weather summary if text is a weather query, else None.
    Tries to extract a city name from the query; falls back to the configured default.
    """
    if not is_weather_query(text):
        return None

    lat, lon, city = default_lat, default_lon, default_city

    m = _CITY_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        # Skip if the candidate looks like a pronoun / stop-word
        if candidate.lower() not in {"the", "a", "an", "my", "your", "our", "it", "this"}:
            geo = _geocode(candidate)
            if geo:
                lat, lon, city = geo

    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "weather_code,wind_speed_10m,precipitation"
            "&temperature_unit=celsius&wind_speed_unit=kmh&timezone=auto"
        )
        data = _fetch(url)
        cur = data["current"]
        temp      = cur["temperature_2m"]
        feels     = cur["apparent_temperature"]
        humidity  = cur["relative_humidity_2m"]
        wind      = cur["wind_speed_10m"]
        precip    = cur["precipitation"]
        condition = _WMO.get(cur["weather_code"], "unknown conditions")

        parts = [f"{city}: {condition}, {temp}°C (feels like {feels}°C)"]
        if precip > 0:
            parts.append(f"precipitation {precip} mm")
        parts.append(f"humidity {humidity}%")
        parts.append(f"wind {wind} km/h")
        return ", ".join(parts)

    except Exception as exc:
        print(f"[weather] fetch failed: {exc}")
        return None
