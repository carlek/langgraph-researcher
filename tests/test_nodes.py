"""Tests for node handlers."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from research_pipeline.nodes import (
    create_analysts,
    human_feedback,
    ask_question,
    search_web,
    search_wikipedia,
    answer_question,
    save_interview,
    write_section,
    write_report,
    write_introduction,
    write_conclusion,
)
from research_pipeline.state import GenerateAnalystsState, InterviewState, ResearchGraphState
from research_pipeline.schemas import Analyst


class TestCreateAnalysts:
    """Test create_analysts node."""

    @patch("research_pipeline.nodes.generate_analyst_personas")
    def test_create_analysts(self, mock_gen_analysts, mock_analyst):
        """Test the create_analysts node."""
        mock_gen_analysts.return_value = [mock_analyst]

        state = {
            "topic": "AI Orchestration",
            "max_analysts": 1,
            "human_analyst_feedback": "",
        }

        result = create_analysts(state)

        assert "analysts" in result
        assert len(result["analysts"]) == 1
        mock_gen_analysts.assert_called_once()

    @patch("research_pipeline.nodes.generate_analyst_personas")
    def test_create_analysts_with_feedback(self, mock_gen_analysts, mock_analyst):
        """Test create_analysts with human feedback."""
        mock_gen_analysts.return_value = [mock_analyst]

        state = {
            "topic": "AI Orchestration",
            "max_analysts": 1,
            "human_analyst_feedback": "Focus on production use cases",
        }

        result = create_analysts(state)

        assert "analysts" in result
        mock_gen_analysts.assert_called_with(
            topic="AI Orchestration",
            feedback="Focus on production use cases",
            n=1,
        )


class TestHumanFeedback:
    """Test human_feedback node."""

    def test_human_feedback_noop(self):
        """Test that human_feedback is a no-op."""
        result = human_feedback({})
        assert result is None


class TestAskQuestion:
    """Test ask_question node."""

    @patch("research_pipeline.nodes.ask_analyst_question")
    def test_ask_question(self, mock_ask, mock_analyst):
        """Test the ask_question node."""
        mock_msg = AIMessage(content="What is LangGraph?")
        mock_ask.return_value = mock_msg

        state = InterviewState(
            analyst=mock_analyst,
            messages=[HumanMessage("Tell me about frameworks")],
        )

        result = ask_question(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "What is LangGraph?"


class TestSearchWeb:
    """Test search_web node."""

    @patch("research_pipeline.nodes.plan_search_query")
    @patch("research_pipeline.nodes.tavily_lookup")
    def test_search_web(self, mock_tavily, mock_plan):
        """Test the search_web node."""
        mock_plan.return_value = "LangGraph vs Temporal"
        mock_tavily.return_value = "<Document>...</Document>"

        state = InterviewState(
            messages=[HumanMessage("Search for comparisons")],
        )

        result = search_web(state)

        assert "context" in result
        assert len(result["context"]) == 1
        mock_plan.assert_called_once()
        mock_tavily.assert_called_once_with("LangGraph vs Temporal")


class TestSearchWikipedia:
    """Test search_wikipedia node."""

    @patch("research_pipeline.nodes.plan_search_query")
    @patch("research_pipeline.nodes.wikipedia_lookup")
    def test_search_wikipedia(self, mock_wiki, mock_plan):
        """Test the search_wikipedia node."""
        mock_plan.return_value = "AI orchestration"
        mock_wiki.return_value = "<Document>...</Document>"

        state = InterviewState(
            messages=[HumanMessage("Search Wikipedia")],
        )

        result = search_wikipedia(state)

        assert "context" in result
        assert len(result["context"]) == 1
        mock_plan.assert_called_once()
        mock_wiki.assert_called_once_with("AI orchestration")


class TestAnswerQuestion:
    """Test answer_question node."""

    @patch("research_pipeline.nodes.answer_as_expert")
    def test_answer_question(self, mock_answer, mock_analyst):
        """Test the answer_question node."""
        mock_msg = AIMessage(content="LangGraph is...")
        mock_msg.name = "expert"
        mock_answer.return_value = mock_msg

        state = InterviewState(
            analyst=mock_analyst,
            messages=[HumanMessage("What is LangGraph?")],
            context=["Retrieved context..."],
        )

        result = answer_question(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        mock_answer.assert_called_once()


class TestSaveInterview:
    """Test save_interview node."""

    def test_save_interview(self):
        """Test the save_interview node."""
        messages = [
            HumanMessage("Question 1"),
            AIMessage("Answer 1"),
            HumanMessage("Question 2"),
            AIMessage("Answer 2"),
        ]

        state = InterviewState(messages=messages)

        result = save_interview(state)

        assert "interview" in result
        assert isinstance(result["interview"], str)
        assert "Question 1" in result["interview"]
        assert "Answer 1" in result["interview"]


class TestWriteSection:
    """Test write_section node."""

    @patch("research_pipeline.nodes.write_interview_section")
    def test_write_section(self, mock_write, mock_analyst):
        """Test the write_section node."""
        mock_write.return_value = "## Section Title\n\nContent here..."

        state = InterviewState(
            analyst=mock_analyst,
            context=["Context 1", "Context 2"],
        )

        result = write_section(state)

        assert "sections" in result
        assert len(result["sections"]) == 1
        assert "## Section Title" in result["sections"][0]
        mock_write.assert_called_once()


class TestWriteReport:
    """Test write_report node."""

    @patch("research_pipeline.nodes.write_report_body")
    def test_write_report(self, mock_write):
        """Test the write_report node."""
        mock_write.return_value = "## Full Report\n\nIntegrated content..."

        state = ResearchGraphState(
            topic="AI Comparison",
            sections=["Section 1", "Section 2"],
        )

        result = write_report(state)

        assert "content" in result
        assert "## Full Report" in result["content"]
        mock_write.assert_called_once()


class TestWriteIntroduction:
    """Test write_introduction node."""

    @patch("research_pipeline.nodes.write_bookend")
    def test_write_introduction(self, mock_write):
        """Test the write_introduction node."""
        mock_write.return_value = "## Introduction\n\nThis report examines..."

        state = ResearchGraphState(
            topic="AI Comparison",
            sections=["Section 1"],
        )

        result = write_introduction(state)

        assert "introduction" in result
        assert "## Introduction" in result["introduction"]
        mock_write.assert_called_once_with(
            "introduction",
            "AI Comparison",
            ["Section 1"],
        )


class TestWriteConclusion:
    """Test write_conclusion node."""

    @patch("research_pipeline.nodes.write_bookend")
    def test_write_conclusion(self, mock_write):
        """Test the write_conclusion node."""
        mock_write.return_value = "## Conclusion\n\nIn summary..."

        state = ResearchGraphState(
            topic="AI Comparison",
            sections=["Section 1"],
        )

        result = write_conclusion(state)

        assert "conclusion" in result
        assert "## Conclusion" in result["conclusion"]
        mock_write.assert_called_once_with(
            "conclusion",
            "AI Comparison",
            ["Section 1"],
        )
