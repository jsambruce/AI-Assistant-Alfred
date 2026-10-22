"""Personal Hugging Face Space for running and submitting Alfred's Unit 4 answers."""
import json
import os
from pathlib import Path
import tempfile

import gradio as gr

from agent import DEFAULT_MODEL
from course_api import get_questions, submit_answers
from evaluation import load_checkpoint, new_checkpoint, rows, run_batch, save_checkpoint


def owner_only(profile):
    if not profile:
        raise gr.Error("Sign in to your Hugging Face account first.")
    space_owner = os.environ.get("SPACE_ID", "").split("/")[0]
    if not space_owner or profile.username != space_owner:
        raise gr.Error("Run this from your own personal Space. Only its owner can use the saved Groq key.")
    return profile.username


def progress_path(state):
    if not state.get("path"):
        folder = Path(tempfile.mkdtemp(prefix="alfred-progress-"))
        state["path"] = str(folder / "unit4-progress.json")
    return state["path"]


def run_ui(count, state, profile: gr.OAuthProfile | None):
    owner_only(profile)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key.startswith("gsk_"):
        raise gr.Error("Add GROQ_API_KEY under this Space's Settings → Secrets.")
    questions = get_questions()
    path = progress_path(state)
    for message, checkpoint in run_batch(state["checkpoint"], path, key, count, questions=questions):
        state["checkpoint"] = checkpoint
        yield message, rows(checkpoint, questions), path if Path(path).exists() else None, state


def restore_ui(file, state, profile: gr.OAuthProfile | None):
    owner_only(profile)
    if not file:
        raise gr.Error("Choose the progress JSON file you downloaded earlier.")
    state["checkpoint"] = load_checkpoint(file, os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    path = progress_path(state)
    save_checkpoint(state["checkpoint"], path)
    return "Progress restored. Completed answers will be skipped.", rows(state["checkpoint"], get_questions()), path, state


def retry_ui(state, profile: gr.OAuthProfile | None):
    owner_only(profile)
    state["checkpoint"]["errors"] = {}
    path = progress_path(state)
    save_checkpoint(state["checkpoint"], path)
    return "Unfinished questions are ready to retry. Click Run next batch after addressing the reported error.", rows(state["checkpoint"], get_questions()), path, state


def submit_ui(confirmed, state, profile: gr.OAuthProfile | None):
    username = owner_only(profile)
    if not confirmed:
        raise gr.Error("Review the answers and tick the submission checkbox first.")
    code_url = f"https://huggingface.co/spaces/{os.environ['SPACE_ID']}/tree/main"
    result = submit_answers(username, code_url, state["checkpoint"], get_questions())
    return json.dumps(result, indent=2)


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Alfred: Unit 4\nRun a few questions, review the answers, then submit them for scoring. Use your own Groq free-plan key. No Hugging Face inference credits are used.")
        gr.LoginButton()
        state = gr.State({"checkpoint": new_checkpoint(os.environ.get("GROQ_MODEL", DEFAULT_MODEL)), "path": None})
        count = gr.Slider(1, 20, value=3, step=1, label="Questions to attempt in this batch")
        run = gr.Button("Run next batch", variant="primary")
        status = gr.Textbox(label="Progress", lines=4, interactive=False)
        table = gr.Dataframe(headers=["Task ID", "Question", "Answer", "Tools used", "Status"], interactive=False, wrap=True)
        download = gr.File(label="Download progress before closing")
        with gr.Accordion("Continue a saved run", open=False):
            upload = gr.File(label="Saved progress JSON", file_types=[".json"], type="filepath")
            restore = gr.Button("Restore progress")
            retry = gr.Button("Retry unfinished questions")
        confirm = gr.Checkbox(label="I reviewed the answers and want to submit them to the course leaderboard.")
        submit = gr.Button("Submit completed answers")
        score = gr.Textbox(label="Official scoring response", lines=7, interactive=False)
        gr.Markdown("Free-plan quotas may interrupt a run. Save your progress and resume later. This version reads text, PDFs, spreadsheets, audio, and available YouTube captions; it cannot interpret images or video scenes. A passing score is not guaranteed.")
        outputs = [status, table, download, state]
        run.click(run_ui, inputs=[count, state], outputs=outputs, concurrency_limit=1, concurrency_id="owner-run")
        restore.click(restore_ui, inputs=[upload, state], outputs=outputs, concurrency_limit=1, concurrency_id="owner-run")
        retry.click(retry_ui, inputs=[state], outputs=outputs, concurrency_limit=1, concurrency_id="owner-run")
        submit.click(submit_ui, inputs=[confirm, state], outputs=score, concurrency_limit=1, concurrency_id="owner-run")
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(share=False)
