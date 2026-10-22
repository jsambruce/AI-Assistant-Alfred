"""Checks for the Unit 4 workflow. No real key, paid calls, or scoring submissions."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from smolagents import Model, ChatMessage, MessageRole
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction
from smolagents.utils import AgentGenerationError

from agent import FreePlanModel, clean_answer, create_alfred, solve
import course_api
from evaluation import new_checkpoint, save_checkpoint, load_checkpoint, run_batch
from research_tools import build_tools, file_in, extract_file, public_url


def question(number=1, filename=""):
    return {"task_id": f"00000000-0000-0000-0000-{number:012d}",
            "question": "Calculate the sum of 2, 3, and 5.", "file_name": filename}


class ScriptedModel(Model):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.saw_observation = False

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            name, arguments = "python_interpreter", {"code": "sum([2, 3, 5])"}
        else:
            self.saw_observation = "10" in str(messages)
            name, arguments = "final_answer", {"answer": "10"}
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[
            ChatMessageToolCall(id=f"call-{self.calls}", type="function",
                                function=ChatMessageToolCallFunction(name=name, arguments=arguments))])


class AgentTests(unittest.TestCase):
    def test_real_framework_loop(self):
        model = ScriptedModel()
        with tempfile.TemporaryDirectory() as folder:
            answer, used = solve(question(), folder, api_key="unused-test-key", model=model)
        self.assertEqual(answer, "10")
        self.assertEqual(used, ["python_interpreter"])
        self.assertTrue(model.saw_observation)

    def test_blank_and_wrong_provider_keys(self):
        for key in ["", "hf_example", "sk_example"]:
            with self.assertRaises(ValueError):
                FreePlanModel(key, delay=0)

    def test_groq_endpoint_and_no_paid_fallback(self):
        model = FreePlanModel("gsk_test_key_not_real", delay=0)
        self.assertEqual(str(model.client.base_url).rstrip("/"), "https://api.groq.com/openai/v1")
        self.assertEqual(model.model_id, "openai/gpt-oss-120b")
        self.assertEqual(model.client.max_retries, 0)

    def test_model_calls_are_paced(self):
        model = FreePlanModel("gsk_test_key_not_real", delay=2)
        model.last_request = 100
        reply = ChatMessage(role=MessageRole.ASSISTANT, content="ok")
        with patch("agent.time.monotonic", side_effect=[100, 102, 102]), patch("agent.time.sleep") as sleep, patch("smolagents.OpenAIModel.generate", return_value=reply):
            self.assertEqual(model.generate([]).content, "ok")
            sleep.assert_called_once_with(1)

    def test_answer_formatting(self):
        self.assertEqual(clean_answer(" FINAL ANSWER: a, b "), "a, b")
        self.assertEqual(clean_answer("```text\n12.50\n```"), "12.50")
        for value in ["", "CANNOT_ANSWER"]:
            with self.assertRaises(ValueError):
                clean_answer(value)

    def test_generation_errors_stop(self):
        class BrokenModel(Model):
            def generate(self, messages, **kwargs):
                raise RuntimeError("test provider failure")
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(AgentGenerationError):
            solve(question(), folder, model=BrokenModel())

    def check_provider_round_trip(self, final_as_error=False):
        import httpx
        from openai import OpenAI
        requests_seen = []
        def respond(request):
            payload = json.loads(request.content)
            requests_seen.append(payload)
            self.assertEqual(payload["tool_choice"], "auto")
            if len(requests_seen) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "calculate-1", "type": "function", "function": {
                        "name": "python_interpreter", "arguments": json.dumps({"code": "17 * 23"})}}]}
            elif final_as_error:
                return httpx.Response(400, json={"error": {"message": "attempted to call tool json",
                    "type": "invalid_request_error", "code": "tool_use_failed",
                    "failed_generation": '{"name":"json","arguments":{"answer":"391"}}'}})
            else:
                message = {"role": "assistant", "content": "391"}
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "openai/gpt-oss-120b", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if len(requests_seen) == 1 else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
        model = FreePlanModel("gsk_test_key_not_real", delay=0)
        model.client = OpenAI(api_key="not-a-real-key", base_url="https://api.groq.com/openai/v1",
                             http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0)
        with tempfile.TemporaryDirectory() as folder:
            answer, used = solve({"question": "Use Python to calculate 17 * 23", "file_name": ""}, folder, model=model)
        self.assertEqual(answer, "391")
        self.assertEqual(used, ["python_interpreter"])
        self.assertEqual(len(requests_seen), 2)
        self.assertIn("391", str(requests_seen[1]["messages"]))
        model.client.close()

    def test_plain_text_final_answer_from_groq(self):
        self.check_provider_round_trip()

    def test_groq_json_tool_error_regression(self):
        self.check_provider_round_trip(final_as_error=True)


class FileTests(unittest.TestCase):
    def test_attachment_cannot_leave_task_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                file_in(folder, "../outside.txt")

    def test_private_network_blocked(self):
        with patch("research_tools.socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 80))]):
            with self.assertRaises(ValueError):
                public_url("http://localhost/secret")

    def test_python_is_read_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.py"
            path.write_text("raise RuntimeError('do not execute')", encoding="utf-8")
            self.assertIn("do not execute", extract_file(path))

    def test_spreadsheet_reader(self):
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as folder:
            book = Workbook()
            book.active.append(["Item", "Sales"])
            book.active.append(["Lunch", 12.5])
            path = Path(folder) / "sales.xlsx"
            book.save(path)
            book.close()
            content = extract_file(path)
            self.assertIn("Lunch", content)
            self.assertIn("12.5", content)

    def test_tool_schemas_and_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "example.txt").write_text("abcdef", encoding="utf-8")
            tools = {t.name: t for t in build_tools(folder, "unused")}
            self.assertEqual(len(tools), 6)
            result = json.loads(tools["read_attachment"](filename="example.txt", offset=3))
            self.assertEqual(result["text"], "def")

    def test_unsupported_image_is_not_faked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "position.png"
            path.write_bytes(b"not-an-image")
            with self.assertRaisesRegex(ValueError, "not supported"):
                extract_file(path)

    @patch("openai.OpenAI")
    def test_audio_uses_groq(self, client):
        client.return_value.audio.transcriptions.create.return_value = SimpleNamespace(text="A sample recording")
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "recording.mp3").write_bytes(b"sample")
            tools = {t.name: t for t in build_tools(folder, "test-key")}
            self.assertEqual(tools["transcribe_audio"](filename="recording.mp3"), "A sample recording")
        self.assertEqual(client.call_args.kwargs["base_url"], "https://api.groq.com/openai/v1")
        self.assertEqual(client.return_value.audio.transcriptions.create.call_args.kwargs["model"], "whisper-large-v3-turbo")


class CourseTests(unittest.TestCase):
    def test_space_code_url_required(self):
        expected = "https://huggingface.co/spaces/example/alfred/tree/main"
        self.assertEqual(course_api.validate_code_url(expected), expected)
        for url in ["https://github.com/example/agent", "https://huggingface.co/spaces/example/alfred", "https://huggingface.co.evil/space"]:
            with self.assertRaises(ValueError):
                course_api.validate_code_url(url)

    @patch("course_api.requests.get")
    def test_official_questions_endpoint(self, get):
        get.return_value.json.return_value = [question()]
        self.assertEqual(course_api.get_questions()[0]["task_id"], question()["task_id"])
        get.assert_called_once_with(course_api.API_URL + "/questions", timeout=30)

    @patch("course_api.requests.get")
    def test_file_download_uses_task_id(self, get):
        response = get.return_value.__enter__.return_value
        response.iter_content.return_value = [b"example"]
        item = question(filename="example.txt")
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(course_api.download_attachment(item, folder).read_bytes(), b"example")
        self.assertEqual(get.call_args.args[0], course_api.API_URL + "/files/" + item["task_id"])

    @patch("huggingface_hub.hf_hub_download")
    @patch("course_api.requests.get")
    def test_missing_course_file_uses_only_original_attachment(self, get, hub_download):
        get.return_value.__enter__.return_value.status_code = 404
        item = question(filename="example.py")
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original.py"
            original.write_text("print(1 + 1)", encoding="utf-8")
            hub_download.return_value = str(original)
            saved = course_api.download_attachment(item, folder)
            self.assertEqual(saved.read_text(), "print(1 + 1)")
        self.assertEqual(hub_download.call_args.kwargs["filename"], "2023/validation/example.py")
        self.assertEqual(hub_download.call_args.kwargs["repo_id"], "gaia-benchmark/GAIA")

    @patch("course_api.requests.post")
    @patch("course_api.requests.get")
    def test_submission_payload(self, get, post):
        get.return_value.json.return_value = {"private": False}
        post.return_value.json.return_value = {"score": 0}
        item = question()
        saved = new_checkpoint()
        saved["answers"][item["task_id"]] = {"answer": "10", "question_hash": course_api.question_hash(item)}
        result = course_api.submit_answers("example", "https://huggingface.co/spaces/example/alfred/tree/main", saved, [item])
        self.assertEqual(result["score"], 0)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload), {"username", "agent_code", "answers"})
        self.assertEqual(payload["answers"], [{"task_id": item["task_id"], "submitted_answer": "10"}])

    @patch("course_api.requests.post")
    @patch("course_api.requests.get")
    def test_changed_question_answer_is_not_submitted(self, get, post):
        get.return_value.json.return_value = {"private": False}
        item = question()
        saved = new_checkpoint()
        saved["answers"][item["task_id"]] = {"answer": "10", "question_hash": "stale"}
        with self.assertRaises(ValueError):
            course_api.submit_answers("example", "https://huggingface.co/spaces/example/alfred/tree/main", saved, [item])
        post.assert_not_called()


class CheckpointTests(unittest.TestCase):
    def test_save_restore(self):
        saved = new_checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            save_checkpoint(saved, path)
            self.assertEqual(load_checkpoint(path), saved)

    @patch("evaluation.download_attachment")
    @patch("evaluation.solve", return_value=("10", ["python_interpreter"]))
    def test_run_saves_and_skips_completed(self, solve_mock, download):
        saved = new_checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            list(run_batch(saved, path, "unused", questions=[question()], model=object()))
            list(run_batch(saved, path, "unused", questions=[question()], model=object()))
            self.assertEqual(load_checkpoint(path)["answers"][question()["task_id"]]["answer"], "10")
        solve_mock.assert_called_once()

    @patch("evaluation.download_attachment")
    @patch("evaluation.solve", side_effect=ValueError("no evidence"))
    def test_unfinished_questions_are_not_answers(self, solve_mock, download):
        saved = new_checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            list(run_batch(saved, Path(folder) / "progress.json", "unused", questions=[question()], model=object()))
        self.assertEqual(saved["answers"], {})
        self.assertIn(question()["task_id"], saved["errors"])

    def test_owner_only_ui(self):
        import app
        with patch.dict(os.environ, {"SPACE_ID": "example/alfred"}):
            self.assertEqual(app.owner_only(SimpleNamespace(username="example")), "example")
            with self.assertRaises(Exception):
                app.owner_only(SimpleNamespace(username="someone-else"))

    @patch("gradio.oauth._get_mocked_oauth_info", return_value={"userinfo": {"preferred_username": "example"}})
    def test_interface_builds(self, oauth):
        import app
        demo = app.build_demo()
        self.assertGreaterEqual(len(demo.config["dependencies"]), 4)
        demo.close()

    @patch("evaluation.download_attachment")
    @patch("evaluation.solve")
    def test_quota_stop_preserves_previous_answer(self, solve_mock, download):
        from smolagents.utils import AgentGenerationError
        import logging
        from smolagents.monitoring import AgentLogger
        logger = AgentLogger()
        solve_mock.side_effect = [("10", []), AgentGenerationError("test quota exhausted", logger)]
        saved = new_checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            events = list(run_batch(saved, path, "unused", max_questions=3,
                                    questions=[question(1), question(2), question(3)], model=object()))
            self.assertEqual(len(load_checkpoint(path)["answers"]), 1)
        self.assertEqual(solve_mock.call_count, 2)
        self.assertIn("Stopped because", events[-1][0])


if __name__ == "__main__":
    unittest.main()
