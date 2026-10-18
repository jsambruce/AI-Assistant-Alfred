"""Check the guestbook and tools without calling an inference provider."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retriever import GuestInfoRetrieverTool, prepare_documents
from tools import HubStatsTool, WeatherInfoTool


class AlfredTests(unittest.TestCase):
    def setUp(self):
        # These are test records, not a replacement for the course dataset.
        self.guests = [
            dict(name="Ada Example", relation="friend", description="Mathematics and computing", email="ada@example.com"),
            dict(name="Niko Example", relation="colleague", description="Wireless energy and pigeons", email="niko@example.com"),
            dict(name="Mia Example", relation="neighbor", description="Painting and music", email="mia@example.com"),
        ]
        self.guestbook = GuestInfoRetrieverTool(prepare_documents(self.guests))

    def test_guest_details(self):
        result = self.guestbook(query="Ada")
        for value in ["Ada Example", "friend", "Mathematics and computing", "ada@example.com"]:
            self.assertIn(value, result)
        self.assertNotIn("Niko Example", result)

    def test_interest(self):
        self.assertIn("Niko Example", self.guestbook(query="pigeons"))

    def test_missing_guest(self):
        self.assertEqual(self.guestbook(query="zzzzunlisted"), "No matching guest information found.")

    @patch("tools.random.choice", return_value={"condition": "Clear", "temp_c": 25})
    def test_weather(self, choose):
        result = WeatherInfoTool()(location="Paris")
        self.assertIn("Simulated weather in Paris: Clear, 25 C", result)
        self.assertIn("not a live forecast", result)

    @patch("tools.list_models")
    def test_hub_statistics(self, list_models):
        list_models.return_value = [SimpleNamespace(id="example/model", downloads=1234)]
        result = HubStatsTool()(author="example")
        self.assertIn("example/model", result)
        self.assertIn("1,234", result)
        list_models.assert_called_once_with(author="example", sort="downloads", limit=1)

    @patch("tools.list_models", return_value=[])
    def test_no_models(self, list_models):
        self.assertIn("No models found", HubStatsTool()(author="example"))

    @patch("tools.list_models", side_effect=ConnectionError)
    def test_hub_failure(self, list_models):
        self.assertIn("Could not get Hub statistics", HubStatsTool()(author="example"))


if __name__ == "__main__":
    unittest.main()
