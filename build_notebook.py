"""Build the Colab notebook from the same files used by the app."""
import json
from pathlib import Path

root = Path(__file__).parent
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


markdown("""# Alfred: Gala Planning Assistant

## Project Overview

I created Alfred to help organize a gala. He can find details about the guests,
check sample weather conditions, search for news, and look up model downloads
on Hugging Face.

I used smolagents to connect the model to these tools. Alfred chooses which tools
to use based on the question and uses the results to put together an answer.

The notebook runs in Google Colab and uses a Hugging Face token for the model.
I kept guest retrieval, the other tools, and the agent setup in separate files.
""")
markdown("""## Install the Libraries

These are the packages I used for the model, guest retrieval, and web search.
""")
requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
code("%pip install -q " + " ".join(f'"{item}"' for item in requirements))
markdown("""## 1. Guest Information

I loaded the guests from `agents-course/unit3-invitees`. Each guest has a document
with their name, relationship to the host, description, and email address.
Their interests are included in the descriptions.

I used `BM25Retriever` to search these documents. The guest tool returns up to
three matches and reports when none of the search words are found.
""")
code("%%writefile retriever.py\n" + (root / "retriever.py").read_text(encoding="utf-8"))
markdown("""## 2. Additional Tools

I added tools for weather, model downloads, and web search. The weather tool picks
from three sample conditions. The Hub tool finds an author's most downloaded
model, and `DuckDuckGoSearchTool` handles searches for public information.
""")
code("%%writefile tools.py\n" + (root / "tools.py").read_text(encoding="utf-8"))
markdown("""## 3. Agent Setup

I used `CodeAgent` with `InferenceClientModel` and connected all four tools.
Alfred can call a tool, read the result, and call another tool if he still needs
information. Once he has enough information, he answers the question.

```text
Question -> Alfred -> tool call -> result -> Alfred -> answer
                         ^                    |
                         |____________________|
                           another tool if needed
```

Alfred decides when to search the guest list, which makes this an agentic RAG
setup. I set `planning_interval=3` so he reviews his plan every three steps and
limited each run to ten steps.
""")
code("%%writefile app.py\n" + (root / "app.py").read_text(encoding="utf-8"))
markdown("""## 4. Connect to the Model

I used Colab Secrets to read `HF_TOKEN`. If the secret is not available, the cell
asks for the token with a hidden input. The token needs Inference Providers
access, and the provider may require credits.
""")
code('''import os
from getpass import getpass

if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        os.environ["HF_TOKEN"] = getpass("Hugging Face token: ")

# Leave this unset to use InferenceClientModel's default model.
# os.environ["HF_MODEL_ID"] = "your-supported-model-id"
''')
markdown("""## 5. Start Alfred

This creates the agent and loads the guest list.
""")
code('''from app import create_alfred

alfred = create_alfred()
print("Alfred is ready.")
''')
markdown("""## 6. Examples

### Guest Lookup

This question asks for Ada Lovelace's details from the guest list.
""")
code('''print(alfred.run(
    "Tell me about Lady Ada Lovelace, including her relationship to the host and email."
))
''')
markdown("""### Weather

This checks the sample weather for Paris. The result can change between runs
because the tool chooses a condition at random.
""")
code('''print(alfred.run("What's the weather like in Paris for the gala?"))
''')
markdown("""### Model Downloads

This asks Alfred to find Qwen's most downloaded model and its download count.
""")
code('''print(alfred.run("What is Qwen's most downloaded model on the Hugging Face Hub?"))
''')
markdown("""### Web Search

This asks for recent news about wireless energy, with links to the sources.
""")
code('''print(alfred.run("Find recent news about wireless energy and include the links."))
''')
markdown("""### Guest Lookup and Web Search

I included this question to see how Alfred uses more than one tool. He needs
Tesla's guest details and recent information about wireless energy to suggest
something to discuss.
""")
code('''print(alfred.run(
    "Use the guestbook to help me prepare a conversation with Dr. Nikola Tesla. "
    "Then look up recent developments in wireless energy and include the links. "
    "Keep private guest details out of the web search."
))
''')
markdown("""### Follow-up Question

I used `reset=False` here so Alfred can refer to the previous answer.
""")
code('''print(alfred.run("Give me two short conversation starters based on that information.", reset=False))
''')
markdown("""## Notes

Weather results are simulated. Web results and model downloads come from live
requests, so they can change. Run the example cells to see the answers.

The guest list comes from the public course dataset. Guest information used in
an answer is sent to the model provider. Keep the token in Secrets and clear
guest details from the outputs before sharing the notebook.

## References

- [Hugging Face Agents Course, Unit 3](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction)
- [Guest retrieval](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees)
- [Tools](https://huggingface.co/learn/agents-course/unit3/agentic-rag/tools)
- [Agent setup](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agent)
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(cells):
    cell["id"] = f"alfred-{i:03d}"
(root / "Alfred_Gala_Agent.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
