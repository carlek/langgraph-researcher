import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import MessagesState

from .schemas import Analyst


class GenerateAnalystsState(TypedDict):
    """State for the analyst-generation loop (create_analysts + human_feedback).

    REPLACED per node:
        topic, max_analysts, human_analyst_feedback, analysts
    """
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]


class InterviewState(MessagesState):
    """Subgraph state for a single analyst interview.

    FAN-IN (reducer = list concat):
        context     — accumulated across search_web + search_wikipedia

    REPLACED per node:
        analyst, interview

    PASSED UP to parent via Send() reducer:
        sections    — appended into ResearchGraphState.sections
    """
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class ResearchGraphState(TypedDict):
    """Parent graph state for the full research pipeline.

    FAN-IN (reducer = list concat):
        sections    — one memo appended per analyst via Send() from conduct_interview

    REPLACED per node:
        topic, max_analysts, human_analyst_feedback, analysts,
        introduction, content, conclusion, final_report
    """
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
