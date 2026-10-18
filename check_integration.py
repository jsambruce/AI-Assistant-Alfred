"""Dependency integration check without provider credentials or network calls."""
import json
from alfred import GuestIndex, build_tools, create_alfred
from smolagents import Model


class UnusedModel(Model):
    def generate(self, messages, **kwargs):
        raise AssertionError("This setup check must not call inference.")


index = GuestIndex([dict(name="Ada Example", relation="friend",
                        description="computing", email="ada@example.com")])
tools = {t.name: t for t in build_tools(index)}
assert set(tools) == {"guest_lookup", "web_search", "weather_info", "hub_stats"}
assert json.loads(tools["guest_lookup"](query="Ada"))["matches"][0]["name"] == "Ada Example"
assert json.loads(tools["weather_info"](location="Paris", event_date="2026-12-01"))["simulated"]
assert json.loads(tools["web_search"](query="ada@example.com"))["status"] == "blocked"
agent = create_alfred(model=UnusedModel(), index=index)
assert set(tools).issubset(agent.tools)
assert agent.max_steps == 8
print("PASS: tool schemas, retrieval, weather, contact blocking and agent construction.")
