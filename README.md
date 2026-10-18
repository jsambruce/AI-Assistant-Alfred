# Alfred: Gala Planning Assistant

## Project Overview

I created Alfred to help organize a gala. He can find details about the guests,
check sample weather conditions, search for news, and look up model downloads
on Hugging Face.

I used smolagents to connect the model to these tools. Alfred chooses which tools
to use based on the question and uses the results to put together an answer.

[Open in Colab](https://colab.research.google.com/github/jsambruce/AI-Assistant-Alfred/blob/main/Alfred_Gala_Agent.ipynb)

## Guest Information

I used `agents-course/unit3-invitees` for the guest list. Each guest is stored in
a LangChain `Document` with their name, relationship to the host, description,
and email. Their interests are included in the descriptions.

`BM25Retriever` searches the documents and returns up to three matches. I added
a check for searches where none of the words match a guest document.

## Tools

I added three tools alongside guest retrieval:

- **Weather:** randomly selects a sample weather condition for a location.
- **Hub statistics:** finds an author's most downloaded model and its download count.
- **Web search:** uses DuckDuckGo to find public information and source links.

The weather is simulated. Hub statistics and search results come from live requests.

## Agent Setup

I connected the tools to a `CodeAgent` using `InferenceClientModel`. Alfred can
call a tool, read the result, and use another tool before answering. He decides
when he needs the guest list, which makes this an agentic RAG setup.

The agent reviews its plan every three steps and has a ten-step limit per run.
Using `reset=False` keeps the previous exchange available for follow-up questions.

The code is split into three files:

- `retriever.py`: loads and searches the guest list.
- `tools.py`: contains the weather and Hub tools and imports web search.
- `app.py`: creates Alfred and runs questions.

## Run the Notebook

Open the Colab link and run the cells in order. Add a Hugging Face token to
Colab Secrets as `HF_TOKEN` and allow notebook access. If the secret is missing,
the notebook asks for the token with a hidden input.

The token needs Inference Providers access. Some providers require credits.
The app uses the default model from `InferenceClientModel`; `HF_MODEL_ID` can be
set to choose another supported model.

## Run Locally

Use Python 3.11 or 3.12 in a virtual environment:

```text
python -m pip install -r requirements.txt
python app.py "Tell me about Lady Ada Lovelace."
```

The app asks for the token if `HF_TOKEN` is not already set.
The package versions in `requirements.txt` are the versions checked with Python 3.12.

## Examples

The notebook includes questions about a guest, weather, model downloads, and
recent news. I also included a question that combines Tesla's guest information
with a search about wireless energy, followed by a question using the same memory.

Run these checks to test the tools without calling the model:

```text
python -m unittest -v
python check_integration.py
```

## Notes

The guest list is the public course dataset. Retrieved guest details are sent
to the model provider. Keep the token out of the files and clear guest details
from notebook outputs before sharing.

The Hub call leaves out the older `direction` argument because current versions
of the library no longer accept it. Sorting by downloads still returns the most
downloaded model first.

## References

- [Hugging Face Agents Course, Unit 3](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction)
- [Guest retrieval](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees)
- [Tools](https://huggingface.co/learn/agents-course/unit3/agentic-rag/tools)
- [Agent setup](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agent)
