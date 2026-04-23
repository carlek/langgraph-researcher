from functools import cache
from typing import Literal
from langchain_anthropic import ChatAnthropic

_MODEL = "claude-sonnet-4-6"

@cache
def llm_for(task: Literal["structured", "interviewer", "writer"]) -> ChatAnthropic:
    cfg = {
        "structured":  dict(temperature=0,   max_tokens=2048),
        "interviewer": dict(temperature=0.3, max_tokens=1024),
        "writer":      dict(temperature=0.2, max_tokens=4096),
    }[task]
    return ChatAnthropic(model=_MODEL, **cfg)
