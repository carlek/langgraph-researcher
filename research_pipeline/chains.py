from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch

from .schemas import Analyst, Perspectives, SearchQuery
from .llm import llm_for
from .prompts import (
    ANALYST_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SECTION_WRITER_INSTRUCTIONS,
    REPORT_WRITER_INSTRUCTIONS,
    INTRO_CONCLUSION_INSTRUCTIONS,    
)
from .observability import traced

def ensure_final_message(
    msgs: list[BaseMessage],
    instruction: str = "Proceed."
) -> list[BaseMessage]:
    """Ensure the message list ends with a user instruction.
    - If last is ToolMessage: leave alone (tool loop in progress).
    - If last is HumanMessage: leave alone (user already spoke).
    - Otherwise append instruction to [msgs] as a human message.
    """
    if not msgs:
        return [HumanMessage(instruction)]
    last = msgs[-1]
    if isinstance(last, (HumanMessage, ToolMessage)):
        return msgs
    return [*msgs, HumanMessage(instruction)]


# @traced("generate_analyst_personas")
def generate_analyst_personas(topic: str, feedback: str, n: int) -> list[Analyst]:
    """Generate N analyst personas for a research topic.

    Structured output via Perspectives wrapper. Feedback is optional guidance
    written by a human between runs of the outer graph.
    """
    sys = SystemMessage(
        ANALYST_INSTRUCTIONS.format(
            topic=topic,
            human_analyst_feedback=feedback or "",
            max_analysts=n,
        )
    )
    chain = llm_for("structured").with_structured_output(Perspectives)
    result = chain.invoke([sys, HumanMessage("Generate the set of analysts.")])
    return result.analysts

# @traced("ask_analyst_question")
def ask_analyst_question(analyst: Analyst, history: list[BaseMessage]) -> AIMessage:
    sys = SystemMessage(QUESTION_INSTRUCTIONS.format(goals=analyst.persona))
    return llm_for("interviewer").invoke(ensure_final_message([sys, *history]))

# @traced("plan_search_query")
def plan_search_query(history: list[BaseMessage]) -> str:
    """Turn the interview so far into a single web-search query."""
    chain = llm_for("structured").with_structured_output(SearchQuery)
    sys = SystemMessage(SEARCH_INSTRUCTIONS)
    return chain.invoke(ensure_final_message([sys, *history])).search_query


# @traced("tavily_lookup")
def tavily_lookup(query: str, k: int = 3) -> str:
    """Run a Tavily web search and format results as <Document> blocks."""
    raw = TavilySearch(max_results=k).invoke({"query": query})
    docs = raw.get("results", []) if isinstance(raw, dict) else raw
    return "\n\n---\n\n".join(
        f'<Document href="{d["url"]}"/>\n{d["content"]}\n</Document>'
        for d in docs
    )

# @traced("wikipedia_lookup")
def wikipedia_lookup(query: str, k: int = 2) -> str:
    """Run a Wikipedia search and format results as <Document> blocks."""
    docs = WikipediaLoader(query=query, load_max_docs=k).load()
    return "\n\n---\n\n".join(
        f'<Document source="{d.metadata["source"]}" page="{d.metadata.get("page","")}"/>\n'
        f'{d.page_content}\n</Document>'
        for d in docs
    )
    
# @traced("answer_as_expert")
def answer_as_expert(
    analyst: Analyst,
    history: list[BaseMessage],
    context: list[str],
) -> AIMessage:
    """Generate the expert's answer to the analyst's latest question.
    Uses only the retrieved context; the prompt forbids outside knowledge.
    Returned message is tagged name='expert' so route_messages can count turns.
    """
    sys = SystemMessage(
        ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
    )
    answer = llm_for("interviewer").invoke(ensure_final_message([sys, *history]))
    answer.name = "expert"
    return answer

# @traced("write_interview_section")
def write_interview_section(analyst: Analyst, context: list[str]) -> str:
    """Write one analyst's memo section from their retrieved sources."""
    sys = SystemMessage(SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description))
    user = HumanMessage(f"Use this source to write your section: {context}")
    return llm_for("writer").invoke([sys, user]).content

# @traced("write_report_body")
def write_report_body(topic: str, sections: list[str]) -> str:
    """Synthesize per-analyst memos into a single consolidated report body."""
    formatted = "\n\n".join(sections)
    sys = SystemMessage(REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted))
    return llm_for("writer").invoke(
        [sys, HumanMessage("Write a report based upon these memos.")]
    ).content
    
# @traced("write_bookend")
def write_bookend(
    kind: Literal["introduction", "conclusion"],
    topic: str,
    sections: list[str],
) -> str:
    instr = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic,
        formatted_str_sections="\n\n".join(sections),
    )
    return llm_for("writer").invoke(
        [SystemMessage(instr), HumanMessage(f"Write the report {kind}")]
    ).content