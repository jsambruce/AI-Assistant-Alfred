"""Run a few questions at a time and save answers before any submission."""
import json
from pathlib import Path
import tempfile

from agent import DEFAULT_MODEL, FreePlanModel, ToolResponseError, solve
from course_api import download_attachment, get_questions, question_hash
from request_budget import RequestSizeError


def new_checkpoint(model_id=DEFAULT_MODEL):
    return {"version": 1, "model": model_id, "answers": {}, "errors": {}}


def save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path, model_id=DEFAULT_MODEL):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("model") != model_id:
        raise ValueError("Use a checkpoint for this version and model.")
    if not isinstance(data.get("answers"), dict) or not isinstance(data.get("errors"), dict):
        raise ValueError("Invalid checkpoint.")
    for task_id, item in data["answers"].items():
        if not isinstance(item, dict) or not isinstance(item.get("answer"), str) or len(item["answer"]) > 20000:
            raise ValueError("Invalid saved answer.")
    return data


def rows(checkpoint, questions):
    output = []
    for question in questions:
        task_id = question["task_id"]
        saved = checkpoint.get("answers", {}).get(task_id)
        valid = saved and saved.get("question_hash") == question_hash(question)
        output.append([task_id, question["question"], saved["answer"] if valid else "",
                       ", ".join(saved.get("tools", [])) if valid else "",
                       "Completed" if valid else checkpoint.get("errors", {}).get(task_id, "Not run")])
    return output


def failure_info(error):
    """Classify the underlying exception without saving provider request data."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    names = {type(item).__name__ for item in chain}
    statuses = {getattr(item, "status_code", None) for item in chain}
    if 413 in statuses or any(isinstance(item, RequestSizeError) for item in chain):
        return "request_size", "The request is too large for the free-plan limit, even after shortening the history. This question remains unfinished.", False
    if 429 in statuses or "RateLimitError" in names:
        return "rate_limit", "Groq's request or token limit was reached. Wait for the quota to reset before retrying.", True
    if statuses & {401, 403} or names & {"AuthenticationError", "PermissionDeniedError"}:
        return "access", "Groq rejected access. Check the saved key and model permissions.", True
    if 402 in statuses:
        return "billing", "The provider rejected the request for billing reasons. Check your account's free-plan access.", True
    for item in chain:
        body = getattr(item, "body", None)
        detail = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(item, ToolResponseError) or (isinstance(detail, dict) and detail.get("code") == "tool_use_failed"):
            return "tool_format", "The model could not produce a valid tool call or answer after correction. Retry this question later.", False
    if "FileNotFoundError" in names:
        return "attachment", "The attachment is unavailable. Check GAIA access and the optional read-access HF_TOKEN.", False
    if "AgentGenerationError" in names:
        return "model_error", "The model request failed. The error does not establish a key or quota problem.", True
    return "task_error", "The question did not finish. Check source access, file support or available evidence.", False


def retry_tool_errors(checkpoint, path, include_request_errors=False):
    """Requeue format failures after an update, preserving answers and a backup."""
    prefixes = ("tool_format:", "AgentGenerationError:")
    if include_request_errors:
        prefixes += ("request_size:", "model_error:")
    retry_ids = [task_id for task_id, message in checkpoint["errors"].items()
                 if isinstance(message, str) and message.startswith(prefixes)]
    if not retry_ids:
        return 0
    path = Path(path)
    backup = path.with_name(path.stem + ".before-tool-retry.json")
    if not backup.exists():
        save_checkpoint(checkpoint, backup)
    for task_id in retry_ids:
        checkpoint["errors"].pop(task_id)
    save_checkpoint(checkpoint, path)
    return len(retry_ids)


def run_batch(checkpoint, path, api_key, max_questions=3, questions=None, model=None):
    questions = questions or get_questions()
    if model is None:
        model = FreePlanModel(api_key, checkpoint["model"])
    attempted = 0
    for question in questions:
        task_id = question["task_id"]
        saved = checkpoint["answers"].get(task_id)
        if (saved and saved.get("question_hash") == question_hash(question)) or task_id in checkpoint["errors"]:
            continue
        if attempted >= max(1, min(int(max_questions), 20)):
            break
        attempted += 1
        yield f"Working on question {questions.index(question) + 1}/{len(questions)}. Free-plan pacing may take several minutes.", checkpoint
        try:
            with tempfile.TemporaryDirectory(prefix="alfred-task-") as directory:
                download_attachment(question, directory)
                answer, tools = solve(question, directory, api_key=api_key, model=model)
            checkpoint["answers"][task_id] = {"question_hash": question_hash(question),
                                              "answer": answer, "tools": tools}
            checkpoint["errors"].pop(task_id, None)
        except Exception as error:
            # Do not store provider exceptions: they can include request details.
            category, message, stop_batch = failure_info(error)
            checkpoint["errors"][task_id] = category + ": " + message
            save_checkpoint(checkpoint, path)
            yield f"Question unfinished: {message} Completed answers were saved.", checkpoint
            if stop_batch:
                yield f"Stopped because: {message}", checkpoint
                return
            continue
        save_checkpoint(checkpoint, path)
        yield f"Saved {len(checkpoint['answers'])}/{len(questions)} answers.", checkpoint
    save_checkpoint(checkpoint, path)
    yield "Batch finished. Review the answers and download the progress file. You can run another batch or submit completed answers.", checkpoint
