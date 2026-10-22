---
title: Alfred Unit 4
emoji: 🕵️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
hf_oauth: true
pinned: false
---

# Alfred: Unit 4

I adapted Alfred from gala planning to research questions. He searches the web,
reads source pages and task files, transcribes audio, and uses a Python tool for
calculations. I kept smolagents for the tool-calling loop and switched inference
to Groq's free plan so this version does not use Hugging Face inference credits.

[Open the notebook in Colab](https://colab.research.google.com/github/jsambruce/AI-Assistant-Alfred/blob/main/unit4/Alfred_Unit4.ipynb)

## What changed

Unit 3 used a private guest list, sample weather, and Hub statistics. Unit 4 uses
general research tools instead. The final answer contains only the requested
result because the course compares answers by exact match. The tool trace keeps
track of the actions used to reach an answer.

The agent uses Groq's `openai/gpt-oss-120b` model. The `openai` package is the API
client used to connect to Groq; no OpenAI account, OpenAI billing, or paid HF
inference is involved. The account must stay on Groq's Free plan. This code cannot
tell whether you have separately upgraded your Groq account, so check your plan
before running it. It does not add billing or fall back to paid services.

## Set up your Hugging Face Space

1. Get a free Groq account and create an API key at https://console.groq.com/keys.
   Do not enable a paid plan for this assignment.
2. Duplicate the [official final assignment template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
   to your personal Hugging Face account. Keep the Space **public** and choose
   the free CPU Basic hardware.
3. Upload the contents of this folder to the Space's root, replacing the template
   files. `app.py`, `README.md`, and `requirements.txt` must be at the root.
4. In the Space's Settings, add a **Secret** named `GROQ_API_KEY` containing your
   Groq key. Do not put it in the code, README, or a public Variable.
5. Wait for the Space to build, then sign in using the Hugging Face button.

Some attachments are currently missing from the course's file endpoint. The
reader falls back to that attachment in the original GAIA dataset. If access is
gated, accept the terms at https://huggingface.co/datasets/gaia-benchmark/GAIA and
add an optional read-access `HF_TOKEN` Secret to the Space. This token is only for
data downloads; it does not need inference permission or paid credits. Without
file access, those questions remain unfinished and text questions can still run.

Only the personal Space's owner can run the agent using its saved key. The
Hugging Face login identifies your submission; it does not pay for inference.

## Run and submit

Start with three questions per batch. The app retrieves the course questions
from `/questions` and downloads attachments through `/files/{task_id}`. It passes
each question and its available file to Alfred. There are no stored answers or
hardcoded task-specific solutions.

Review the answer and tools-used columns after a batch. Download the progress JSON
before closing the page. To continue after a restart, upload that JSON under
Continue a saved run, restore it, and run the next batch. Completed answers are
skipped. Failed questions are listed separately and are never submitted as error
strings. Use Retry unfinished questions after resolving a problem or waiting for
your quota to reset.

When ready, tick the review checkbox and click Submit completed answers. The
app sends your signed-in username, the public Space code link ending in
`/tree/main`, and the generated `{task_id, submitted_answer}` entries to `/submit`.
Generating answers alone does not submit anything. Submitting partial answers is
possible, but it does not establish a passing score. Use the official returned
score as the result.

Unit 4 uses 20 questions and the introduction sets a passing target of 30%.
After receiving a qualifying score, claim the certificate on the
[certificate page](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate).
Writing or uploading this project does not itself earn the certificate.

## Free-plan limits and file support

Groq lists request and token limits for the free plan. The model waits at least
65 seconds between chat requests to reduce short-term rate-limit failures. A full
run can take a long time and may need to be split across quota resets. The wait
does not guarantee you will stay under a token or daily limit. Audio transcription
has its own quota. Completed answers are saved before a stopped run.

The readers support text, Python source, CSV, JSON, PDFs, and XLSX spreadsheets.
The Python tool restricts imports and execution time; downloaded source is read
as text rather than automatically executed. Audio uses Groq's Whisper API.
YouTube captions may be unavailable or blocked by the video host. Captions cannot
answer questions about visible scenes. Image interpretation and visual chess
positions are not supported in this version. Unsupported questions can remain
unanswered; a 30% score has not been measured or guaranteed.

Question text, downloaded excerpts, and audio used by the agent are sent to Groq.
Web search uses a public search engine. Keep real private data out of this project.
Progress files are local to the running Space until you download them; free
Spaces do not provide durable storage by default.

## Files

- `agent.py`: Groq connection, tool-calling loop, and final-answer formatting.
- `research_tools.py`: search, pages, attachments, audio, captions, and Python.
- `course_api.py`: official question, file, and submission requests.
- `evaluation.py`: batches, progress files, and unfinished questions.
- `app.py`: Hugging Face login, run controls, and explicit submission.
- `Alfred_Unit4.ipynb`: Colab setup and a small test before deploying the Space.

Run `python -m unittest -v` after installing `requirements.txt` to check the
workflow without a real Groq key. Tests use synthetic fixtures and scripted model
responses; they do not establish a live score or model-answer accuracy.

## References

- [Unit 4 introduction](https://huggingface.co/learn/agents-course/en/unit4/introduction)
- [Hands-on instructions](https://huggingface.co/learn/agents-course/en/unit4/hands-on)
- [Official template code](https://huggingface.co/spaces/agents-course/Final_Assignment_Template/blob/main/app.py)
- [Certificate instructions](https://huggingface.co/learn/agents-course/en/unit4/get-your-certificate)
- [Groq free-plan rate limits](https://console.groq.com/docs/rate-limits)
- [Groq models](https://console.groq.com/docs/models)
- [Groq API compatibility](https://console.groq.com/docs/openai)
- [Original GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
