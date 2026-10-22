"""Keep research history small enough for Groq's free-plan request limits."""
from copy import deepcopy
from functools import lru_cache
import json

INPUT_TOKEN_BUDGET = 4800
OUTPUT_TOKEN_BUDGET = 2048
TRIM_NOTICE = "\n[Excerpt shortened for the request limit. Read the source again if needed.]\n"


class RequestSizeError(ValueError):
    """A question cannot fit the request budget without losing essential input."""


@lru_cache(maxsize=1)
def encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_harmony")
    except Exception:
        # No tokenizer/network access: use a more conservative byte count.
        return None


def token_count(text):
    tokenizer = encoding()
    return len(tokenizer.encode(text, disallowed_special=())) if tokenizer else len(text.encode("utf-8"))


def input_size(request):
    """Estimate text/schema tokens and reserve extra space for provider formatting."""
    messages, tools = request.get("messages", []), request.get("tools", [])
    text = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, separators=(",", ":"))
    return token_count(text) + 256 + 16 * len(messages) + 32 * len(tools)


def shorten_text(text, limit):
    if token_count(text) <= limit:
        return text
    low, high = 0, len(text)
    best = TRIM_NOTICE
    while low <= high:
        count = (low + high) // 2
        head = count * 2 // 3
        tail = count - head
        candidate = text[:head] + TRIM_NOTICE + (text[-tail:] if tail else "")
        if token_count(candidate) <= limit:
            best = candidate
            low = count + 1
        else:
            high = count - 1
    return best


def shorten_observation(content, limit):
    """Retain source metadata and valid JSON when trimming page/file excerpts."""
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return shorten_text(content, limit)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        text = data.pop("text")
        data["excerpt_shortened"] = True
        overhead = token_count(json.dumps(data, ensure_ascii=False)) + 16
        data["text"] = shorten_text(text, max(32, limit - overhead))
        return json.dumps(data, ensure_ascii=False)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Search results keep their full URLs and titles.
        for item in data:
            if isinstance(item.get("snippet"), str):
                item["snippet"] = shorten_text(item["snippet"], 32)
        return json.dumps(data, ensure_ascii=False)
    return shorten_text(content, limit)


def fit_request(request, budget=INPUT_TOKEN_BUDGET):
    """Trim old observations, then old complete tool pairs; keep the question."""
    request = deepcopy(request)
    messages = request["messages"]
    if input_size(request) <= budget:
        return request
    observations = [m for m in messages if m.get("role") == "tool"]
    # Keep the newest evidence longest. Never edit function arguments or IDs.
    for floor in (512, 160):
        for message in observations:
            while input_size(request) > budget:
                original = message.get("content", "")
                size = token_count(original)
                if size <= floor:
                    break
                reduced = shorten_observation(original, max(floor, size // 2))
                if token_count(reduced) >= size:
                    break
                message["content"] = reduced
    # Drop only complete old call/result pairs, retaining the latest pair.
    while input_size(request) > budget:
        pairs = [i for i in range(len(messages) - 1)
                 if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")
                 and messages[i + 1].get("role") == "tool"
                 and len(messages[i]["tool_calls"]) == 1
                 and messages[i]["tool_calls"][0]["id"] == messages[i + 1].get("tool_call_id")]
        if len(pairs) <= 1:
            raise RequestSizeError("The question, tool definitions and latest result cannot fit the free-plan request budget.")
        del messages[pairs[0]:pairs[0] + 2]
    return request
