"""Build the Colab notebook from the Unit 4 files."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


markdown("""# Alfred: Unit 4

## Project Overview

I adapted Alfred to answer research questions for the final assessment. I kept
smolagents and replaced the gala tools with web research, file readers, audio
transcription, captions, and calculations.

This version connects directly to Groq's free plan. It does not use Hugging Face
inference credits. A free Groq account and API key are still needed. Free limits
can interrupt a run, so answers are saved and can be resumed later.

This notebook lets me test the agent before uploading it to a public Hugging Face
Space. The Space handles Hugging Face sign-in and the official score submission.
""")
markdown("""## 1. Install the Libraries

Run these cells in a fresh CPU runtime. The model runs at Groq, so a paid Colab
GPU is not needed. Keep your Groq account on the Free plan.
""")
requirements = (ROOT / "requirements.txt").read_text().splitlines()
code("%pip install -q " + " ".join(f'"{item}"' for item in requirements))
markdown("""## 2. Create the Project Files

The agent chooses tools, reads the results, and continues until it can answer or
reaches its step limit. It returns only the requested result for exact-match
scoring. Tool calls are saved separately from the final answer.

```text
Course question + optional attachment
                 |
                 v
             Alfred (Groq)
                 |
                 v
     Search / pages / files / audio / Python
                 |
                 +------> Alfred -----> answer
```

The progress runner keeps generated answers. Sending them to the leaderboard
is a separate action in the Space.
""")
for name in ["course_api.py", "research_tools.py", "agent.py", "evaluation.py", "app.py", "test_unit4.py"]:
    code(f"%%writefile {name}\n" + (ROOT / name).read_text(encoding="utf-8"))
code("%%writefile requirements.txt\n" + (ROOT / "requirements.txt").read_text())
code("%%writefile README.md\n" + (ROOT / "README.md").read_text(encoding="utf-8"))
markdown("""## 3. Add the Groq Key

Create a key at https://console.groq.com/keys. In Colab Secrets, save it as
`GROQ_API_KEY` and enable notebook access. This cell reads the saved key each time
it runs, so an old environment value does not override a changed secret.
Do not paste the key into the notebook code or share it in chat.
""")
code('''import os
from google.colab import userdata

os.environ.pop("GROQ_API_KEY", None)
key = userdata.get("GROQ_API_KEY").strip()
if not key.startswith("gsk_"):
    raise ValueError("Save your full Groq API key, starting with gsk_, in Colab Secrets.")
os.environ["GROQ_API_KEY"] = key
print("Groq key loaded.")

# Optional: this token downloads gated GAIA files; it is not used for inference.
os.environ.pop("HF_TOKEN", None)
try:
    file_token = userdata.get("HF_TOKEN").strip()
except Exception:
    file_token = ""
if file_token:
    if not file_token.startswith("hf_"):
        raise ValueError("The optional HF_TOKEN must be a Hugging Face read token starting with hf_.")
    os.environ["HF_TOKEN"] = file_token
''')
markdown("""## 4. Check the Tools and Agent Loop

These tests use sample data and scripted model responses. They check the tools,
loop, progress files, and submission format without using the Groq key. They do
not prove that the live model will pass the evaluation.
""")
code('''import unittest
suite = unittest.defaultTestLoader.discover(".", pattern="test_unit4.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
''')
markdown("""## 5. Try the Live Model

This simple calculation checks the real Groq connection and the Python tool.
Use the single code cell below. No diagnostic or patch cells are needed.
If it fails with a rate limit, keep your key and wait for the quota to reset.
There is no automatic switch to a paid provider.
""")
code('''from pathlib import Path
import importlib
import agent

# Pick up the latest agent.py if the project-writing cells were rerun.
importlib.invalidate_caches()
importlib.reload(agent)
from agent import solve
print("Alfred version:", agent.AGENT_VERSION)

Path("downloads").mkdir(exist_ok=True)
sample = {"question": "Use the Python tool to calculate 17 * 23. Return only the integer.",
          "file_name": ""}
answer, used_tools = solve(sample, "downloads")
print("Answer:", answer)
print("Tools:", used_tools)
assert answer == "391", "Review the model's answer before running evaluation questions."
assert "python_interpreter" in used_tools, "The model did not use the Python tool."
print("Alfred test passed.")
''')
markdown("""## 6. Run a Small Evaluation Batch

This cell retrieves the current course questions and tries three unfinished
questions. It saves progress to `unit4-progress.json` after each question. Rerun
the cell to continue. Failed questions are kept out of the answer list; use the
Space's retry control when the source or quota issue has been resolved.
Tool-format failures leave that question unfinished while the batch continues.
The status message distinguishes these failures from key-access or rate limits.

The 65-second spacing between model requests reduces rate-limit errors, so this
can take several minutes. It does not remove daily or token quotas.

Some course files are missing from the file endpoint. The reader tries the
original GAIA attachment instead. If it is gated, accept the dataset's access
terms and optionally add a read-access `HF_TOKEN` Secret. That token is for file
downloads, not inference. Questions without accessible files stay unfinished.
""")
code('''from evaluation import new_checkpoint, load_checkpoint, run_batch

progress_file = Path("unit4-progress.json")
checkpoint = load_checkpoint(progress_file) if progress_file.exists() else new_checkpoint()
for status, checkpoint in run_batch(checkpoint, progress_file, os.environ["GROQ_API_KEY"], max_questions=3):
    print(status)
print("Saved answers:", len(checkpoint["answers"]))
print("Unfinished:", len(checkpoint["errors"]))
''')
markdown("""## 7. Save Progress

Download the file before closing Colab. It contains generated answers and tool
names, not the API key. To resume, upload it into the notebook's working directory
or restore it in the Space.
""")
code('''from google.colab import files
files.download("unit4-progress.json")
''')
markdown("""## 8. Publish and Submit

Duplicate the [official template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
to your personal account as a public Gradio Space on free CPU Basic hardware.
Upload this project's files to the Space root and save `GROQ_API_KEY` as a Secret
in its Settings. The README enables the Hugging Face login button.

Sign in to your Space, restore progress if needed, and continue the batches.
Review the answers, tick the checkbox, and click Submit completed answers.
The app sends your username, public Space code link ending in `/tree/main`, and
generated answers to the official scoring endpoint. This notebook does not
automatically submit answers.

The course introduction sets a target of at least 30% on 20 questions. Only an
official score shows whether the agent passed. After qualifying, use the
[certificate page](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate).

## Limits

The model can be wrong. Some sites block requests, transcripts can be missing,
and this version cannot interpret images or count objects in video frames.
Audio is sent to Groq for transcription; read excerpts are sent to the model.
No score or certificate is promised by this notebook.

## References

- [Unit 4 hands-on](https://huggingface.co/learn/agents-course/en/unit4/hands-on)
- [Unit 4 introduction](https://huggingface.co/learn/agents-course/en/unit4/introduction)
- [Groq free-plan limits](https://console.groq.com/docs/rate-limits)
- [Groq API](https://console.groq.com/docs/openai)
""")
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}, "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"unit4-{index:03d}"
(ROOT / "Alfred_Unit4.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
