from langgraph.graph import END, START, StateGraph
from .nodes import (
    # interview subgraph
    ask_question,
    search_web,
    search_wikipedia,
    answer_question,
    save_interview,
    write_section,
    route_messages,
    # parent graph
    create_analysts,
    human_feedback,
    write_report,
    write_introduction,
    write_conclusion,
    finalize_report,
    initiate_all_interviews,
)
from .state import InterviewState, ResearchGraphState

class N:
    CREATE_ANALYSTS   = "create_analysts"
    HUMAN_FEEDBACK    = "human_feedback"
    ASK_QUESTION      = "ask_question"
    SEARCH_WEB        = "search_web"
    SEARCH_WIKIPEDIA  = "search_wikipedia"
    ANSWER_QUESTION   = "answer_question"
    SAVE_INTERVIEW    = "save_interview"
    WRITE_SECTION     = "write_section"
    CONDUCT_INTERVIEW = "conduct_interview"
    WRITE_REPORT      = "write_report"
    WRITE_INTRO       = "write_introduction"
    WRITE_CONCLUSION  = "write_conclusion"
    FINALIZE          = "finalize_report"


def build_interview_graph() -> StateGraph:
    """Build (but do not compile) the per-analyst interview subgraph."""

    # ---------- interview subgraph ----------
    # Contract with parent:
    #   reads:  analyst (from Send payload), messages (seeded by parent)
    #   writes: sections (fanned in via parent's Annotated[list, operator.add])

    g = StateGraph(InterviewState)
    g.add_node(N.ASK_QUESTION, ask_question)
    g.add_node(N.SEARCH_WEB, search_web)
    g.add_node(N.SEARCH_WIKIPEDIA, search_wikipedia)
    g.add_node(N.ANSWER_QUESTION, answer_question)
    g.add_node(N.SAVE_INTERVIEW, save_interview)
    g.add_node(N.WRITE_SECTION, write_section)

    g.add_edge(START, N.ASK_QUESTION)   
    g.add_edge(N.ASK_QUESTION, N.SEARCH_WEB)
    g.add_edge(N.ASK_QUESTION, N.SEARCH_WIKIPEDIA)
    g.add_edge(N.SEARCH_WEB, N.ANSWER_QUESTION)
    g.add_edge(N.SEARCH_WIKIPEDIA, N.ANSWER_QUESTION)
    g.add_conditional_edges(N.ANSWER_QUESTION, 
                                    route_messages, 
                                    [N.ASK_QUESTION, N.SAVE_INTERVIEW])
    g.add_edge(N.SAVE_INTERVIEW, N.WRITE_SECTION)
    g.add_edge(N.WRITE_SECTION, END)
    return g


def build_research_graph(checkpointer=None):
    """Build and compile the full research pipeline.
    Checkpointer strategy — three run modes, one graph:
    * LangGraph Studio / `langgraph dev`: pass `None` (the default). The
      platform injects its own checkpointer and ignores any you supply.
    * CLI / programmatic via the driver: driver passes `InMemorySaver()`
      so graph.update_state / resume work between stream() calls.
    * Production: caller passes SqliteSaver / PostgresSaver for durability
    """
    
    # ---------- parent research graph ----------

    g = StateGraph(ResearchGraphState)
    g.add_node(N.CREATE_ANALYSTS, create_analysts)
    g.add_node(N.HUMAN_FEEDBACK, human_feedback)
    g.add_node(N.CONDUCT_INTERVIEW, build_interview_graph().compile())
    g.add_node(N.WRITE_REPORT, write_report)
    g.add_node(N.WRITE_INTRO, write_introduction)
    g.add_node(N.WRITE_CONCLUSION, write_conclusion)
    g.add_node(N.FINALIZE, finalize_report)

    g.add_edge(START, N.CREATE_ANALYSTS)
    g.add_edge(N.CREATE_ANALYSTS, N.HUMAN_FEEDBACK)
    g.add_conditional_edges(
        N.HUMAN_FEEDBACK, 
        initiate_all_interviews,
        [N.CREATE_ANALYSTS, N.CONDUCT_INTERVIEW],
    )
    g.add_edge(N.CONDUCT_INTERVIEW, N.WRITE_REPORT)
    g.add_edge(N.CONDUCT_INTERVIEW, N.WRITE_INTRO)
    g.add_edge(N.CONDUCT_INTERVIEW, N.WRITE_CONCLUSION)
    g.add_edge(
        [N.WRITE_CONCLUSION, N.WRITE_REPORT, N.WRITE_INTRO],
        N.FINALIZE,
    )
    g.add_edge(N.FINALIZE, END)

    compile_kwargs = {"interrupt_before": [N.HUMAN_FEEDBACK]}
    
    # Only add a checkpointer if it's a real saver instance.
    # Studio injects its own persistence/checkpointer causing a TypeError.
    if hasattr(checkpointer, "put") and hasattr(checkpointer, "get_tuple"):
        compile_kwargs["checkpointer"] = checkpointer
    
    return g.compile(**compile_kwargs)
