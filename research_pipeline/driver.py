"""Driver for the research pipeline.

Encapsulates the HITL loop that the graph's `interrupt_before=['human_feedback']`
requires: run until interrupt, collect feedback, resume; if feedback wasn't
'approve', the graph loops back to create_analysts and pauses again.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import Callable
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from .graph import build_research_graph, N
from .schemas import Analyst


FeedbackFn = Callable[[list[Analyst]], str]

def auto_approve(_: list[Analyst]) -> str:
    """Default feedback strategy: accept whatever was generated."""
    return "approve"

def interactive_feedback(analysts: list[Analyst]) -> str:
    """Print analysts to stderr and read feedback from stdin."""
    print("\n=== Generated Analysts ===", file=sys.stderr)
    for i, a in enumerate(analysts, 1):
        print(f"\n[{i}] {a.name} — {a.role}", file=sys.stderr)
        print(f"    Affiliation: {a.affiliation}", file=sys.stderr)
        print(f"    {a.description}", file=sys.stderr)
    print(file=sys.stderr)
    try:
        raw = input("Feedback (empty/'approve' to continue): ").strip()
    except EOFError:
        raw = ""
    return raw or "approve"

def run_research(
    topic: str,
    max_analysts: int = 3,
    feedback_fn: FeedbackFn = auto_approve,
    thread_id: str | None = None,
    checkpointer=None,
    max_feedback_rounds: int = 5,
) -> str:
    """Run the pipeline end-to-end and return the final report markdown."""
    graph = build_research_graph(checkpointer=checkpointer or InMemorySaver())
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial = {"topic": topic, "max_analysts": max_analysts}
    _drain(graph.stream(initial, config))

    for _ in range(max_feedback_rounds):
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        if N.HUMAN_FEEDBACK not in snapshot.next:
            raise RuntimeError(
                f"Unexpected pause at {snapshot.next!r}; expected human_feedback"
            )
        analysts = snapshot.values.get("analysts", [])
        feedback = feedback_fn(analysts)
        graph.update_state(config, {"human_analyst_feedback": feedback})
        _drain(graph.stream(None, config))
    else:
        raise RuntimeError(
            f"Exceeded {max_feedback_rounds} feedback rounds without 'approve'"
        )

    final_state = graph.get_state(config)
    report = final_state.values.get("final_report")
    if not report:
        raise RuntimeError("Graph terminated without producing final_report")
    return report

def _drain(stream) -> None:
    for _ in stream:
        pass


# ---------- CLI ----------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research_pipeline",
        description="Run the multi-analyst research pipeline.",
    )
    p.add_argument("topic", help="Research topic")
    p.add_argument("-n", "--max-analysts", type=int, default=3,
                   help="Number of analyst personas (default: 3)")
    p.add_argument("--interactive", action="store_true",
                   help="Prompt for feedback (default: auto-approve)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Write report to file; default: stdout")
    p.add_argument("--thread-id", type=str, default=None,
                   help="Resume an existing thread")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable INFO-level chain tracing to stderr")
    return p

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    feedback_fn = interactive_feedback if args.interactive else auto_approve
    load_dotenv()
    report = run_research(
        topic=args.topic,
        max_analysts=args.max_analysts,
        feedback_fn=feedback_fn,
        thread_id=args.thread_id,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())