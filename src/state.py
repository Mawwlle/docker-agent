from typing import TypedDict
from langchain_core.messages import BaseMessage


class State(TypedDict, total=False):
    messages: list[BaseMessage]
    plan: list[str]
    draft: list[str]
    result: str | None
