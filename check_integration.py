"""Check that the course tools can be attached to a CodeAgent."""
from smolagents import CodeAgent, Model

from app import create_alfred
from retriever import GuestInfoRetrieverTool, prepare_documents


class UnusedModel(Model):
    def generate(self, messages, **kwargs):
        raise AssertionError("This check should not call inference.")


guestbook = GuestInfoRetrieverTool(prepare_documents([
    dict(name="Ada Example", relation="friend", description="computing", email="ada@example.com")
]))
alfred = create_alfred(model=UnusedModel(), guest_info_tool=guestbook)
assert isinstance(alfred, CodeAgent)
assert {"guest_info_retriever", "weather_info", "hub_stats", "web_search"}.issubset(alfred.tools)
assert alfred.planning_interval == 3
assert alfred.max_steps == 10
print("All four tools are connected to the CodeAgent.")
