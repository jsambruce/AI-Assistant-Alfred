"""Run a few questions at a time and save answers before any submission."""
import json
from pathlib import Path
import tempfile

from agent import DEFAULT_MODEL, FreePlanModel, solve
from course_api import download_attachment, get_questions, question_hash


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
            message = "task did not finish. Check the key, limits, source access, or file support."
            if isinstance(error, FileNotFoundError):
                message = "attachment unavailable from the course/GAIA. Check GAIA access and optional read-access HF_TOKEN."
            checkpoint["errors"][task_id] = type(error).__name__ + ": " + message
            save_checkpoint(checkpoint, path)
            yield "This question did not finish. Completed answers were saved.", checkpoint
            cause = error
            while cause is not None:
                if type(cause).__name__ in {"AgentGenerationError", "RateLimitError", "AuthenticationError", "PermissionDeniedError"}:
                    yield "Stopped because the model could not respond. Check Groq key/model access or wait for the free quota to reset, then use Retry unfinished questions.", checkpoint
                    return
                cause = cause.__cause__ or cause.__context__
            continue
        save_checkpoint(checkpoint, path)
        yield f"Saved {len(checkpoint['answers'])}/{len(questions)} answers.", checkpoint
    save_checkpoint(checkpoint, path)
    yield "Batch finished. Review the answers and download the progress file. You can run another batch or submit completed answers.", checkpoint
