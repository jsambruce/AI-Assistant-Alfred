"""Tools for web search, sample weather, and model downloads."""
import random

from huggingface_hub import list_models
from smolagents import DuckDuckGoSearchTool, Tool


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Return simulated weather for a location. This is sample data, not a live forecast."
    inputs = {"location": {"type": "string", "description": "City where the gala is taking place."}}
    output_type = "string"

    def forward(self, location: str) -> str:
        # Pick a sample weather condition for the location.
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20},
        ]
        weather = random.choice(weather_conditions)
        return (
            f"Simulated weather in {location}: {weather['condition']}, "
            f"{weather['temp_c']} C. This is not a live forecast."
        )


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Find an author's most downloaded model on the Hugging Face Hub."
    inputs = {"author": {"type": "string", "description": "Hugging Face username or organization."}}
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            # Current Hub versions sort downloads descending without a direction argument.
            models = list(list_models(author=author, sort="downloads", limit=1))
            if not models:
                return f"No models found for {author}."
            model = models[0]
            downloads = f"{model.downloads:,}" if model.downloads is not None else "unavailable"
            return (
                f"Most downloaded model by {author}: {model.id}. "
                f"Downloads reported by the Hub: {downloads} (last 30 days). "
                f"Source: https://huggingface.co/{model.id}"
            )
        except Exception as error:
            return f"Could not get Hub statistics for {author}: {type(error).__name__}."
