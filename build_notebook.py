"""Assemble a portable notebook with embedded application source."""
import json
from pathlib import Path

root = Path(__file__).parent
cells = []


def markdown(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=text.splitlines(True)))


markdown("""# Alfred — Gala Agent (Agentic RAG)
Based on [Hugging Face Agents Course Unit 3](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction).
Run cells in order in Colab. This notebook is self-contained; no repository clone is needed.
It includes official guest retrieval, public search, simulated weather and live Hub stats.
Guest records are loaded at runtime. Do not publish tokens or confidential outputs.
""")
markdown("## 1. Install dependencies\nUse a fresh CPU runtime. No GPU is required; model inference runs through Hugging Face.")
code('%pip install -q "smolagents>=1.24,<2" "datasets>=3,<5" "huggingface-hub>=0.34,<2" "ddgs>=9,<10"')
markdown("""## 2. Implementation and architecture
`Question → model → selected tool → observation → model → final answer`.
The model can repeat tool calls and combine evidence, with an eight-step budget.
`ToolCallingAgent` uses structured tool calls; it does not execute model-written Python.
Guest retrieval uses BM25 over all four official fields. Only positive matches are
returned, with row references; interests are supported by description text.
Web results have URLs, Hub counts have retrieval times, and weather is explicitly simulated.
""")
code((root / "alfred.py").read_text(encoding="utf-8").split('if __name__ == "__main__":')[0])
markdown("## 3. Offline checks\nSynthetic fixtures test behavior; they are never used as the application dataset.")
code((root / "test_alfred.py").read_text(encoding="utf-8").replace('from alfred import GuestIndex, simulated_weather\n', '').split('if __name__ == "__main__":')[0] + '\nresult = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RetrievalTests))\nassert result.wasSuccessful()\n')
markdown("""## 4. Credentials
Create a Hugging Face token with Inference Providers permission. Store it in Colab
Secrets as `HF_TOKEN` with notebook access, or enter it in the hidden prompt.
Provider credits may be required. Change the model ID if the default is unavailable;
the selected model must support tool calling. Never paste a token into source code.
""")
code('''from getpass import getpass
if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        os.environ["HF_TOKEN"] = getpass("Hugging Face token: ")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
''')
markdown("## 5. Load the official guest data and assemble Alfred\nNo private dataset records are printed in this setup cell.")
code('''from datasets import load_dataset
guest_index = GuestIndex(load_dataset(DATASET_ID, split="train"))
alfred = create_alfred(index=guest_index)
print("Alfred ready. Tools:", list(alfred.tools))
''')
markdown("## 6. Try each capability\nEach run starts a fresh agent memory. These calls consume inference credits.")
code('''questions = [
    "Tell me about Lady Ada Lovelace from the guestbook, including her relationship to the host.",
    "Give the simulated weather in Paris on 2026-12-01. Label it as simulation.",
    "Find Qwen's most downloaded models and give the metric period and source links.",
    "Find recent public research news on wireless energy and cite the sources.",
]
for question in questions:
    print("QUESTION:", question)
    print(alfred.run(question))
    print()
''')
markdown("""## 7. Multi-tool evaluation
The following request needs guest retrieval, public research, weather and Hub stats.
Inspect actual tool names below. Missing required calls are a failed routing evaluation,
even if the answer sounds plausible. The trace shows actions, not private model reasoning.
""")
code('''question = (
    "Prepare a host briefing about Dr. Nikola Tesla using the guestbook. "
    "Find recent public wireless-energy research, without putting private guest details into search. "
    "Also get Qwen's top models and simulated weather for Paris on 2026-12-01. "
    "Include source references and distinguish simulation from live data."
)
answer = alfred.run(question)
print(answer)
called = [call.name for step in alfred.memory.steps
          for call in (getattr(step, "tool_calls", None) or [])]
print("Tools called:", called)
required = {"guest_lookup", "web_search", "weather_info", "hub_stats"}
print("Required tools missing:", sorted(required - set(called)))
''')
markdown("""## 8. Review and submit
- Check the briefing against retrieved records and URLs; do not accept invented guest facts.
- Try an unknown guest and an invalid weather date. Alfred should acknowledge missing evidence.
- Retrieval is lexical; retry keywords for paraphrases. Search/provider failures are possible.
- The guest data is sent to the inference provider. This teaching notebook assumes an authorized host;
  it does not implement authentication or production confidentiality controls.
- Clear all cell outputs before publishing. Keep tokens in Secrets.
- Publish these files in your public GitHub repository and submit that repository URL.
  Or save a Colab copy and enable **Anyone with the link → Viewer**.

See the [course tools](https://huggingface.co/learn/agents-course/unit3/agentic-rag/tools)
and [agent integration](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agent).
""")
notebook = dict(cells=cells, metadata={"colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}}, nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell["id"] = f"alfred-{i:03d}"
(root / "Alfred_Gala_Agent.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
