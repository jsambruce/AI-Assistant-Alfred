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

from agent import FreePlanModel, ToolResponseError, argument_problem, clean_answer, create_alfred, solve
import course_api
from evaluation import new_checkpoint, save_checkpoint, load_checkpoint, run_batch, failure_info, retry_tool_errors
from research_tools import build_tools, file_in, extract_file, public_url
from request_budget import INPUT_TOKEN_BUDGET, OUTPUT_TOKEN_BUDGET, RequestSizeError, fit_request, input_size, token_count


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
            self.assertNotIn("stop", payload)
            self.assertNotIn("max_tokens", payload)
            self.assertEqual(payload["max_completion_tokens"], OUTPUT_TOKEN_BUDGET)
            self.assertNotIn("include_reasoning", payload)
            self.assertNotIn("final_answer", [tool["function"]["name"] for tool in payload["tools"]])
            if len(requests_seen) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "calculate-1", "type": "function", "function": {
                        "name": "python_interpreter", "arguments": json.dumps({"code": "17 * 23"})}}]}
            elif final_as_error and len(requests_seen) == 2:
                return httpx.Response(400, json={"error": {"message": "attempted to call tool json",
                    "type": "invalid_request_error", "code": "tool_use_failed",
                    "failed_generation": '{"name": "answer", "arguments": 999"}'}})
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
        self.assertEqual(len(requests_seen), 3 if final_as_error else 2)
        history = requests_seen[1]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "tool"])
        self.assertEqual(history[-1]["tool_call_id"], "calculate-1")
        self.assertIn("391", history[-1]["content"])
        self.assertEqual(json.loads(history[-2]["tool_calls"][0]["function"]["arguments"]), {"code": "17 * 23"})
        if final_as_error:
            self.assertNotIn("999", str(requests_seen[2]["messages"]))
            self.assertEqual(requests_seen[2]["messages"][-1]["role"], "user")
        model.client.close()

    def test_plain_text_final_answer_from_groq(self):
        self.check_provider_round_trip()

    def test_groq_json_tool_error_regression(self):
        self.check_provider_round_trip(final_as_error=True)

    def test_truncated_empty_response_is_not_an_answer(self):
        model = FreePlanModel("gsk_test_key_not_real", delay=0)
        reply = ChatMessage(role=MessageRole.ASSISTANT, content=None,
                            raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]))
        with tempfile.TemporaryDirectory() as folder:
            final = create_alfred(folder, model=ScriptedModel()).tools["final_answer"]
            with patch("smolagents.OpenAIModel.generate", return_value=reply) as generate:
                with self.assertRaisesRegex(ValueError, "output limit"):
                    model.generate([], tools_to_call_from=[final], stop_sequences=["Observation:"])
                self.assertNotIn("stop_sequences", generate.call_args.kwargs)


class NativeProtocolTests(unittest.TestCase):
    """Exercise real framework execution and SDK serialization, with fake HTTP only."""

    def run_protocol(self, replies, expected=None, tasks=1):
        import httpx
        from openai import OpenAI
        seen = []

        def respond(request):
            payload = json.loads(request.content)
            seen.append(payload)
            self.assertEqual(request.url.host, "api.groq.com")
            self.assertLessEqual(input_size(payload), INPUT_TOKEN_BUDGET)
            self.assertLess(input_size(payload) + payload["max_completion_tokens"], 8000)
            self.assertNotIn("final_answer", [t["function"]["name"] for t in payload["tools"]])
            pending = []
            for message in payload["messages"]:
                if pending:
                    self.assertEqual(message["role"], "tool")
                    self.assertEqual(message["tool_call_id"], pending.pop(0))
                if message.get("tool_calls"):
                    for call in message["tool_calls"]:
                        self.assertIsInstance(json.loads(call["function"]["arguments"]), dict)
                        pending.append(call["id"])
            self.assertFalse(pending)
            status, message = replies.pop(0)
            if status != 200:
                return httpx.Response(status, json=message)
            return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
                "model": "openai/gpt-oss-120b", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        model = FreePlanModel("gsk_test_key_not_real", delay=0)
        model.client = OpenAI(api_key="fixture", base_url="https://api.groq.com/openai/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0)
        try:
            with tempfile.TemporaryDirectory() as folder:
                for index in range(tasks):
                    sample = {"question": f"Task {index}: use Python for the calculation.", "file_name": ""}
                    if expected is None:
                        with self.assertRaises(AgentGenerationError):
                            solve(sample, folder, model=model)
                    else:
                        answer, used = solve(sample, folder, model=model)
                        self.assertEqual(answer, expected)
                        self.assertTrue(used)
        finally:
            model.client.close()
        self.assertFalse(replies, "Expected requests were not made")
        return seen

    def tool_reply(self, code, call_id="python-1", name="python_interpreter"):
        return (200, {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps({"code": code})}}]})

    def text_reply(self, text):
        return (200, {"role": "assistant", "content": text})

    def test_two_real_python_steps(self):
        seen = self.run_protocol([self.tool_reply("7 * 8"), self.tool_reply("56 + 9", "python-2"),
                                  self.text_reply("65")], expected="65")
        self.assertIn("56", seen[1]["messages"][-1]["content"])
        self.assertIn("65", seen[2]["messages"][-1]["content"])
        self.assertEqual([m["role"] for m in seen[2]["messages"]],
                         ["system", "user", "assistant", "tool", "assistant", "tool"])

    def test_tool_exception_sent_as_tool_result(self):
        seen = self.run_protocol([self.tool_reply("1 / 0"), self.tool_reply("7 * 8", "python-2"),
                                  self.text_reply("56")], expected="56")
        self.assertIn("error", seen[1]["messages"][-1]["content"].lower())

    def test_empty_response_gets_one_repair(self):
        seen = self.run_protocol([self.tool_reply("9 * 9"), self.text_reply(""), self.text_reply("81")], expected="81")
        self.assertEqual(seen[2]["messages"][-1]["role"], "user")
        self.assertIn("81", seen[2]["messages"][-2]["content"])

    def test_repeated_empty_response_stops(self):
        self.run_protocol([self.tool_reply("9 * 9"), self.text_reply(""), self.text_reply("")])

    def test_repeated_malformed_response_stops_without_using_error_answer(self):
        error = (400, {"error": {"message": "Invalid call", "type": "invalid_request_error",
            "code": "tool_use_failed", "failed_generation": '{"name":"json","arguments":{"answer":"999"}}'}})
        self.run_protocol([self.tool_reply("9 * 9"), error, error])

    def test_quota_error_is_not_retried(self):
        self.run_protocol([(429, {"error": {"message": "Quota exhausted", "type": "rate_limit_error"}})])

    def test_unknown_function_is_not_executed(self):
        self.run_protocol([self.tool_reply("1", name="unknown"), self.tool_reply("1", name="unknown")])

    def test_new_question_does_not_reuse_previous_memory(self):
        seen = self.run_protocol([self.tool_reply("7 * 8"), self.text_reply("56"),
                                  self.tool_reply("7 * 8"), self.text_reply("56")], expected="56", tasks=2)
        self.assertEqual([m["role"] for m in seen[2]["messages"]], ["system", "user"])
        self.assertNotIn("Task 0:", str(seen[2]["messages"]))

    def function_reply(self, name, arguments, call_id):
        return (200, {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]})

    @patch("research_tools.fetch_page")
    @patch("ddgs.DDGS")
    def test_search_read_and_pagination_after_exact_missing_url_error(self, search, fetch):
        url = "https://example.org/source"
        search.return_value.text.return_value = [{"title": "Fixture source", "href": url, "body": "A reference"}]
        fetch.return_value = (url, ("<p>" + "x" * 3600 + " Reference value: 42.</p>").encode(), "text/html")
        malformed = (400, {"error": {
            "message": "Tool call validation failed: parameters for read_webpage did not match schema: missing properties: url",
            "type": "invalid_request_error", "code": "tool_use_failed",
            "failed_generation": '{"name":"read_webpage","arguments":{"cursor":0,"loc":3000}}'}})
        seen = self.run_protocol([
            self.function_reply("web_search", {"query": "reference value"}, "search-1"),
            malformed,
            self.function_reply("read_webpage", {"url": url, "offset": 0}, "page-1"),
            self.function_reply("read_webpage", {"url": url, "offset": 3500}, "page-2"),
            self.text_reply("42")], expected="42")
        note = seen[2]["messages"][-1]["content"]
        self.assertIn("Missing required arguments: url", note)
        self.assertIn("Unrecognized arguments: cursor, loc", note)
        self.assertIn("url: string (required)", note)
        self.assertIn("offset: integer (optional)", note)
        first_page = json.loads(seen[3]["messages"][-1]["content"])
        self.assertEqual(first_page["next_call"], {"url": url, "offset": 3500})
        self.assertIn("Reference value: 42", seen[4]["messages"][-1]["content"])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], [url, url])

    @patch("research_tools.fetch_page")
    def test_invalid_arguments_in_success_response_are_not_executed(self, fetch):
        malformed = self.function_reply("read_webpage", {"cursor": 0, "loc": 3000}, "bad-1")
        seen = self.run_protocol([malformed, malformed])
        fetch.assert_not_called()
        self.assertIn("Missing required arguments: url", seen[1]["messages"][-1]["content"])

    def test_argument_types_and_optional_offsets(self):
        tools = {tool.name: tool for tool in build_tools(".", "unused")}
        reader = tools["read_webpage"]
        self.assertEqual(argument_problem(reader, {"url": "https://example.org"}), "")
        for args in [{"url": 1}, {"url": "https://example.org", "offset": True},
                     {"url": "https://example.org", "offset": "3500"},
                     {"url": "https://example.org", "offset": None}]:
            self.assertTrue(argument_problem(reader, args))

    def test_413_retries_with_a_smaller_request(self):
        error = (413, {"error": {"message": "Request too large: Limit 8000, Requested 8185",
                                 "type": "tokens", "code": "rate_limit_exceeded"}})
        seen = self.run_protocol([self.tool_reply("list(range(3000))"), error, self.text_reply("2999")], expected="2999")
        self.assertGreater(input_size(seen[1]), input_size(seen[2]))
        self.assertLessEqual(input_size(seen[2]), int(INPUT_TOKEN_BUDGET * 0.65))
        self.assertIn("shortened", seen[2]["messages"][-1]["content"])

    def test_repeated_413_is_bounded(self):
        error = (413, {"error": {"message": "Request too large", "type": "tokens", "code": "rate_limit_exceeded"}})
        self.run_protocol([error, error])


class RequestBudgetTests(unittest.TestCase):
    def history(self, count=8):
        messages = [{"role": "system", "content": "Research using the tools."},
                    {"role": "user", "content": "Find the reference value; preserve this question."}]
        for i in range(count):
            url = f"https://example.org/page-{i}"
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [{"id": f"page-{i}", "type": "function",
                    "function": {"name": "read_webpage", "arguments": json.dumps({"url": url, "offset": 3500})}}]},
                {"role": "tool", "tool_call_id": f"page-{i}", "content": json.dumps({
                    "source": url, "offset": 3500, "total_characters": 100000,
                    "next_call": {"url": url, "offset": 7000},
                    "text": " ".join(str(n) for n in range(2000))})}])
        return {"messages": messages, "tools": [], "max_completion_tokens": OUTPUT_TOKEN_BUDGET}

    def test_long_history_fits_without_mutating_question_or_memory(self):
        original = self.history()
        snapshot = json.loads(json.dumps(original))
        result = fit_request(original)
        self.assertGreater(input_size(original), INPUT_TOKEN_BUDGET)
        self.assertLessEqual(input_size(result), INPUT_TOKEN_BUDGET)
        self.assertEqual(original, snapshot)
        self.assertEqual(result["messages"][:2], original["messages"][:2])
        self.assertEqual(result["messages"][-1]["tool_call_id"], "page-7")
        for i in range(2, len(result["messages"]), 2):
            call, observation = result["messages"][i:i+2]
            self.assertEqual(call["tool_calls"][0]["id"], observation["tool_call_id"])
            metadata = json.loads(observation["content"])
            self.assertEqual(metadata["source"], json.loads(call["tool_calls"][0]["function"]["arguments"])["url"])
            self.assertEqual(metadata["next_call"]["offset"], 7000)

    def test_long_tool_arguments_drop_only_complete_old_pairs(self):
        request = self.history(count=5)
        for message in request["messages"][2:-2:2]:
            message["tool_calls"][0]["function"]["arguments"] = json.dumps({"code": " ".join(str(n) for n in range(2000))})
        result = fit_request(request)
        self.assertLess(len(result["messages"]), len(request["messages"]))
        self.assertEqual(result["messages"][-2]["tool_calls"][0]["id"], "page-4")
        self.assertEqual(result["messages"][-1]["tool_call_id"], "page-4")

    def test_irreducible_question_stops_before_provider_call(self):
        model = FreePlanModel("gsk_fixture", delay=0)
        with patch.object(model.client.chat.completions, "create") as create:
            with self.assertRaises(RequestSizeError):
                model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": " ".join(str(n) for n in range(12000))}])])
            create.assert_not_called()
        model.client.close()

    def test_repair_instruction_and_unicode_survive(self):
        request = self.history(count=3)
        request["messages"][-1]["content"] = "你好 🌍 " * 5000
        request["messages"].append({"role": "user", "content": "Supply the required url and offset."})
        result = fit_request(request)
        self.assertEqual(result["messages"][-1], request["messages"][-1])
        self.assertLessEqual(input_size(result), INPUT_TOKEN_BUDGET)

    def test_byte_fallback_still_enforces_limit(self):
        with patch("request_budget.encoding", return_value=None):
            result = fit_request(self.history(count=2))
            self.assertLessEqual(input_size(result), INPUT_TOKEN_BUDGET)
            self.assertEqual(token_count("你好"), len("你好".encode("utf-8")))


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
    @patch("evaluation.download_attachment")
    @patch("evaluation.solve")
    def test_tool_failure_continues_batch_without_saving_failed_answer(self, solve_mock, download):
        from smolagents.monitoring import AgentLogger
        wrapper = AgentGenerationError("fixture tool failure", AgentLogger())
        wrapper.__cause__ = ToolResponseError("invalid read_webpage arguments")
        solve_mock.side_effect = [wrapper, ("10", ["python_interpreter"])]
        saved = new_checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            events = list(run_batch(saved, path, "unused", max_questions=2,
                                    questions=[question(1), question(2)], model=object()))
            restored = load_checkpoint(path)
        self.assertEqual(solve_mock.call_count, 2)
        self.assertNotIn(question(1)["task_id"], restored["answers"])
        self.assertTrue(restored["errors"][question(1)["task_id"]].startswith("tool_format:"))
        self.assertEqual(restored["answers"][question(2)["task_id"]]["answer"], "10")
        messages = " ".join(event[0] for event in events)
        self.assertNotIn("quota", messages)
        self.assertNotIn("key", messages)

    def test_api_failures_use_actual_status(self):
        import httpx
        from openai import APIStatusError, BadRequestError, RateLimitError, AuthenticationError
        from smolagents.monitoring import AgentLogger
        for status, error_type, body, expected, stop in [
            (400, BadRequestError, {"code": "tool_use_failed"}, "tool_format", False),
            (400, BadRequestError, {"code": "invalid_request_error"}, "model_error", True),
            (401, AuthenticationError, {}, "access", True),
            (413, APIStatusError, {"code": "rate_limit_exceeded"}, "request_size", False),
            (429, RateLimitError, {}, "rate_limit", True)]:
            response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
            underlying = error_type("private request details", response=response, body=body)
            wrapper = AgentGenerationError("fixture", AgentLogger())
            wrapper.__cause__ = underlying
            category, message, stopped = failure_info(wrapper)
            self.assertEqual((category, stopped), (expected, stop))
            self.assertNotIn("private request details", message)

    def test_retry_preserves_answers_unrelated_errors_and_backup(self):
        saved = new_checkpoint()
        saved["answers"][question(1)["task_id"]] = {"answer": "10"}
        saved["errors"] = {"old": "AgentGenerationError: old status", "new": "tool_format: failed", "file": "attachment: missing"}
        original = json.loads(json.dumps(saved))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            self.assertEqual(retry_tool_errors(saved, path), 2)
            self.assertEqual(load_checkpoint(path)["answers"], original["answers"])
            self.assertEqual(saved["errors"], {"file": "attachment: missing"})
            backup = path.with_name("progress.before-tool-retry.json")
            self.assertEqual(load_checkpoint(backup), original)
            self.assertEqual(retry_tool_errors(saved, path), 0)

    def test_size_update_requeues_legacy_error_and_preserves_access_failures(self):
        saved = new_checkpoint()
        saved["errors"] = {"legacy": "model_error: unclassified", "size": "request_size: too big", "key": "access: denied"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(retry_tool_errors(saved, Path(folder) / "progress.json", include_request_errors=True), 2)
            self.assertEqual(saved["errors"], {"key": "access: denied"})

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
