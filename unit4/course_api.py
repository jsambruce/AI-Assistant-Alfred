"""The question, attachment, and scoring endpoints used by the course template."""
import hashlib
import re
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

API_URL = "https://agents-course-unit4-scoring.hf.space"


def question_hash(question):
    text = question["question"] + "\n" + question.get("file_name", "")
    return hashlib.sha256(text.encode()).hexdigest()


def get_questions():
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()
    if not isinstance(questions, list) or not questions:
        raise ValueError("The course did not return a question list.")
    ids = set()
    for item in questions:
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("A question is missing its text.")
        task_id = item.get("task_id", "")
        if not re.fullmatch(r"[0-9a-f-]{36}", task_id) or task_id in ids:
            raise ValueError("Invalid or duplicate task ID.")
        ids.add(task_id)
    return questions


def download_attachment(question, directory):
    name = question.get("file_name", "")
    if not name:
        return None
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Invalid attachment filename.")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    # The endpoint takes the task ID, not the attachment's filename.
    with requests.get(f"{API_URL}/files/{question['task_id']}", timeout=60, stream=True) as response:
        if response.status_code == 404:
            # Some course tasks list a filename but the scoring service has no file mapping.
            # Fetch only that attachment, never GAIA's answer metadata.
            from huggingface_hub import hf_hub_download
            try:
                source = hf_hub_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset",
                    filename="2023/validation/" + name, token=os.environ.get("HF_TOKEN") or False)
                data = Path(source).read_bytes()
                if len(data) > 25 * 1024 * 1024:
                    raise ValueError("Attachment exceeds the 25 MB limit.")
                target.write_bytes(data)
                return target
            except Exception as error:
                raise FileNotFoundError(
                    "The course file endpoint has no attachment for this task. "
                    "For the GAIA fallback, accept access at https://huggingface.co/datasets/gaia-benchmark/GAIA "
                    "and optionally add a read-access HF_TOKEN Secret. This does not use inference credits."
                ) from error
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > 25 * 1024 * 1024:
                raise ValueError("Attachment exceeds the 25 MB limit.")
        target.write_bytes(data)
    return target


def validate_code_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment:
        raise ValueError("Use your public Hugging Face Space code URL.")
    if not re.fullmatch(r"/spaces/[\w.-]+/[\w.-]+/tree/main/?", parsed.path):
        raise ValueError("The code URL must look like https://huggingface.co/spaces/USER/SPACE/tree/main")
    return url.rstrip("/")


def submit_answers(username, code_url, checkpoint, questions):
    code_url = validate_code_url(code_url)
    if not username.strip():
        raise ValueError("Sign in to Hugging Face before submitting.")
    # Check that the submitted Space can be read publicly.
    space_id = "/".join(urlparse(code_url).path.split("/")[2:4])
    public = requests.get(f"https://huggingface.co/api/spaces/{space_id}", timeout=20)
    public.raise_for_status()
    if public.json().get("private", True):
        raise ValueError("The Space must be public for the course submission.")
    answers = []
    for item in questions:
        saved = checkpoint.get("answers", {}).get(item["task_id"])
        if saved and saved.get("question_hash") == question_hash(item) and saved.get("answer", "").strip():
            answers.append({"task_id": item["task_id"], "submitted_answer": saved["answer"]})
    if not answers:
        raise ValueError("There are no completed answers to submit.")
    response = requests.post(f"{API_URL}/submit", json={
        "username": username.strip(), "agent_code": code_url, "answers": answers,
    }, timeout=90)
    response.raise_for_status()
    return response.json()
