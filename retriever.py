"""Search the guest list for names, relationships, and interests."""
import re

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

DATASET_ID = "agents-course/unit3-invitees"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def prepare_documents(guests):
    # Store each guest's details in one document.
    return [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]),
            metadata={"name": guest["name"]},
        )
        for guest in guests
    ]


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Look up gala guests by name, relationship, or interests. "
        "Returns their biography, relationship to the host, and email."
    )
    inputs = {"query": {"type": "string", "description": "Guest name or keywords to look up."}}
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        if not docs:
            raise ValueError("The guest list is empty.")
        self.retriever = BM25Retriever.from_documents(docs, k=3, preprocess_func=tokenize)

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)
        # BM25 can return documents even when none of the query words match.
        words = set(tokenize(query))
        results = [doc for doc in results if words.intersection(tokenize(doc.page_content))]
        if not results:
            return "No matching guest information found."
        return "\n\n".join(doc.page_content for doc in results)


def load_guest_dataset():
    guests = load_dataset(DATASET_ID, split="train")
    return GuestInfoRetrieverTool(prepare_documents(guests))
