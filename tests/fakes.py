"""Fakes shared by the processor/routing tests: an LLM that returns a
fixed reply or calls search_documents with given arguments."""
from unittest.mock import MagicMock


class FakeLLMResult:
    def __init__(self, content, total_tokens=42, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = {"total_tokens": total_tokens}


def fake_llm(content=None, total_tokens=42, error=None, tool_args=None):
    """`tool_args` set -> the model "calls" search_documents with them."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm  # route_message binds the search tool first
    if error:
        llm.invoke.side_effect = error
    else:
        tool_calls = [{"name": "search_documents", "args": tool_args, "id": "t1"}] if tool_args else None
        llm.invoke.return_value = FakeLLMResult("" if tool_args else content, total_tokens, tool_calls)
    return llm
