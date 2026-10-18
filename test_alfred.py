"""Offline checks; synthetic fixtures never replace official data in the application."""
import unittest
from alfred import GuestIndex, simulated_weather


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = GuestIndex([
            dict(name="Ada Example", relation="friend", description="Mathematics and computing", email="ada@example.com"),
            dict(name="Niko Example", relation="colleague", description="Wireless energy and pigeons", email="niko@example.com"),
        ])

    def test_name_and_email(self):
        result = self.index.search("Ada")[0]
        self.assertEqual(result["email"], "ada@example.com")
        self.assertTrue(result["source"].endswith("/row/0"))

    def test_interest(self):
        self.assertEqual(self.index.search("pigeons wireless")[0]["name"], "Niko Example")

    def test_missing_guest(self):
        self.assertEqual(self.index.search("zzzzunlisted"), [])
        self.assertEqual(self.index.search(""), [])

    def test_schema(self):
        with self.assertRaises(ValueError):
            GuestIndex([{"name": "incomplete"}])

    def test_simulation_is_stable_and_labelled(self):
        a = simulated_weather("Paris", "2026-12-01")
        self.assertEqual(a, simulated_weather("Paris", "2026-12-01"))
        self.assertTrue(a["simulated"])

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            simulated_weather("Paris", "tomorrow")


if __name__ == "__main__":
    unittest.main()
