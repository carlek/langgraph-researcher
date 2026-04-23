from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.constants import Send
from .chains import (
    generate_analyst_personas,
    ask_analyst_question,
    plan_search_query,
    tavily_lookup,
    wikipedia_lookup,
    answer_as_expert,
    write_interview_section, 
    write_report_body,
    write_bookend,
)
from .state import GenerateAnalystsState, InterviewState, ResearchGraphState


# ---------- analyst generation / HITL node ----------

def create_analysts(state: GenerateAnalystsState) -> dict:
    """Generate analyst personas from topic + optional human feedback."""
    analysts = generate_analyst_personas(
        topic=state["topic"],
        feedback=state.get("human_analyst_feedback", ""),
        n=state["max_analysts"],
    )
    return {"analysts": analysts}

def human_feedback(state) -> None:
    """No-op. Exists solely as an interrupt target.

    Graph is compiled with interrupt_before=['human_feedback'], so execution
    pauses here. The caller then uses graph.update_state(config, {...}) to
    write 'human_analyst_feedback' into state before resuming.
    """
    pass


# ---------- interview subgraph nodes ----------

def ask_question(state: InterviewState) -> dict:
    msg = ask_analyst_question(state["analyst"], state["messages"])
    return {"messages": [msg]}

def search_web(state: InterviewState) -> dict:
    query = plan_search_query(state["messages"])
    return {"context": [tavily_lookup(query)]}

def search_wikipedia(state: InterviewState) -> dict:
    query = plan_search_query(state["messages"])
    return {"context": [wikipedia_lookup(query)]}

def answer_question(state: InterviewState) -> dict:
    msg: AIMessage = answer_as_expert(
        analyst=state["analyst"],
        history=state["messages"],
        context=state["context"],
    )
    return {"messages": [msg]}

def save_interview(state: InterviewState) -> dict:
    """Flatten the message history into a single transcript string."""
    return {"interview": get_buffer_string(state["messages"])}

def write_section(state: InterviewState) -> dict:
    section = write_interview_section(
        analyst=state["analyst"],
        context=state["context"],
    )
    return {"sections": [section]}


# ---------- parent graph writing nodes ----------

def write_report(state: ResearchGraphState) -> dict:
    return {"content": write_report_body(state["topic"], state["sections"])}

def write_introduction(state) -> dict:
    return {"introduction": write_bookend("introduction", state["topic"], state["sections"])}

def write_conclusion(state) -> dict:
    return {"conclusion": write_bookend("conclusion",   state["topic"], state["sections"])}

def finalize_report(state: ResearchGraphState) -> dict:
    """Stitch intro + body + conclusion, lifting any Sources block to the end.

    Bug fixes vs. original:
      - str.strip("## Insights") → removeprefix (strip was operating on a
        character set, not a substring).
      - bare except → except ValueError for the split path.
    """
    content = state["content"].removeprefix("## Insights").lstrip()
    sources = None
    if "\n## Sources\n" in content:
        try:
            content, sources = content.split("\n## Sources\n", 1)
        except ValueError:
            sources = None

    final = (
        state["introduction"]
        + "\n\n---\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )
    if sources:
        final += "\n\n## Sources\n" + sources
    return {"final_report": final}

# ---------- conditional routers ----------

# strings (not N.*) to avoid a circular import
# - graph.py imports from nodes so nodes can't import from graph. 
# - These must match those in graph.py.
_ASK_QUESTION = "ask_question"
_SAVE_INTERVIEW = "save_interview"
_CREATE_ANALYSTS   = "create_analysts"
_CONDUCT_INTERVIEW = "conduct_interview"

def route_messages(state: InterviewState, name: str = "expert") -> str:
    """Decide whether to ask another question or end the interview.

    Ends when either:
      - the expert has answered >= max_num_turns times, or
      - the analyst's last question contains the sign-off phrase.
    """
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    expert_answers = sum(
        1 for m in messages if isinstance(m, AIMessage) and m.name == name
    )
    if expert_answers >= max_num_turns:
        return _SAVE_INTERVIEW

    # messages[-2] is the analyst's last question
    # messages[-1] is the expert answer that just triggered this router
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return _SAVE_INTERVIEW

    # continue asking 
    return _ASK_QUESTION

def initiate_all_interviews(state: ResearchGraphState):
    """Route after human_feedback interrupt.

    - feedback == 'approve' (or missing): fan out via Send() to interview subgraph
    - otherwise: loop back to create_analysts with feedback as guidance
    """
    feedback = state.get("human_analyst_feedback", "approve")
    if feedback.lower() != "approve":
        return _CREATE_ANALYSTS

    topic = state["topic"]
    return [
        Send(
            _CONDUCT_INTERVIEW,
            {
                "analyst": analyst,
                "messages": [HumanMessage(
                    content=f"So you said you were writing an article on {topic}?"
                )],
            },
        )
        for analyst in state["analysts"]
    ]
    