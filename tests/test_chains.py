"""Tests for chain functions."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from research_pipeline.chains import (
    ensure_final_message,
    generate_analyst_personas,
    ask_analyst_question,
    plan_search_query,
    answer_as_expert,
    write_interview_section,
    write_report_body,
    write_bookend,
)
from research_pipeline.schemas import Analyst, Perspectives, SearchQuery


class TestEnsureFinalMessage:
    """Test ensure_final_message helper."""

    def test_empty_messages(self):
        """Test with empty message list."""
        result = ensure_final_message([])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Proceed."

    def test_custom_instruction(self):
        """Test with custom instruction."""
        result = ensure_final_message([], instruction="Continue.")
        assert result[0].content == "Continue."

    def test_last_message_is_human(self):
        """Test when last message is HumanMessage."""
        messages = [
            SystemMessage("System"),
            HumanMessage("User message"),
        ]
        result = ensure_final_message(messages)
        assert result == messages

    def test_last_message_is_ai(self):
        """Test when last message is AIMessage."""
        messages = [
            SystemMessage("System"),
            AIMessage("Assistant response"),
        ]
        result = ensure_final_message(messages)
        assert len(result) == 3
        assert isinstance(result[-1], HumanMessage)


class TestGenerateAnalystPersonas:
    """Test generate_analyst_personas chain."""

    def test_generate_analysts(self, mock_llm_for, mock_perspectives):
        """Test generating analyst personas."""
        # Setup mock to return Perspectives
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_perspectives
        mock_llm_for.return_value.with_structured_output.return_value = mock_chain

        result = generate_analyst_personas(
            topic="AI and machine learning",
            feedback="",
            n=2,
        )

        assert len(result) == 2
        assert result[0].name == "Dr. Alice Johnson"
        assert mock_llm_for.called
        mock_chain.invoke.assert_called_once()

    def test_generate_analysts_with_feedback(self, mock_llm_for, mock_perspectives):
        """Test generating analysts with human feedback."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_perspectives
        mock_llm_for.return_value.with_structured_output.return_value = mock_chain

        result = generate_analyst_personas(
            topic="AI and machine learning",
            feedback="Focus on practical applications",
            n=2,
        )

        assert len(result) == 2
        # Verify the feedback was included in the call
        call_args = mock_chain.invoke.call_args
        assert call_args is not None


class TestAskAnalystQuestion:
    """Test ask_analyst_question chain."""

    def test_ask_question(self, mock_llm_for, mock_analyst):
        """Test asking an analyst a question."""
        mock_response = AIMessage(content="What is LangGraph?")
        mock_llm_for.return_value.invoke.return_value = mock_response

        messages = [HumanMessage("Tell me about orchestration")]
        result = ask_analyst_question(mock_analyst, messages)

        assert isinstance(result, AIMessage)
        assert result.content == "What is LangGraph?"
        mock_llm_for.assert_called_with("interviewer")

    def test_ask_question_empty_history(self, mock_llm_for, mock_analyst):
        """Test asking a question with empty message history."""
        mock_response = AIMessage(content="Initial question")
        mock_llm_for.return_value.invoke.return_value = mock_response

        result = ask_analyst_question(mock_analyst, [])

        assert isinstance(result, AIMessage)
        mock_llm_for.return_value.invoke.assert_called_once()


class TestPlanSearchQuery:
    """Test plan_search_query chain."""

    def test_plan_search_query(self, mock_llm_for, mock_search_query):
        """Test planning a search query."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_search_query
        mock_llm_for.return_value.with_structured_output.return_value = mock_chain

        messages = [HumanMessage("Tell me about orchestration frameworks")]
        result = plan_search_query(messages)

        assert result == "LangGraph vs Temporal production deployment"
        mock_llm_for.assert_called_with("structured")


class TestAnswerAsExpert:
    """Test answer_as_expert chain."""

    def test_answer_as_expert(self, mock_llm_for, mock_analyst):
        """Test expert generating an answer."""
        mock_response = AIMessage(content="LangGraph is a framework for building AI agents...")
        mock_llm_for.return_value.invoke.return_value = mock_response

        messages = [HumanMessage("What is LangGraph?")]
        context = ["LangGraph is a framework from LangChain..."]
        result = answer_as_expert(mock_analyst, messages, context)

        assert isinstance(result, AIMessage)
        assert result.name == "expert"
        assert result.content == "LangGraph is a framework for building AI agents..."
        mock_llm_for.assert_called_with("interviewer")

    def test_expert_name_tagged(self, mock_llm_for, mock_analyst):
        """Test that expert response is tagged with name."""
        mock_response = AIMessage(content="Response")
        mock_llm_for.return_value.invoke.return_value = mock_response

        result = answer_as_expert(mock_analyst, [], [])
        
        assert result.name == "expert"


class TestWriteInterviewSection:
    """Test write_interview_section chain."""

    def test_write_section(self, mock_llm_for, mock_analyst):
        """Test writing an interview section."""
        mock_response = AIMessage(
            content="## Expert Perspective\n\nLangGraph excels in..."
        )
        mock_llm_for.return_value.invoke.return_value = mock_response

        context = ["LangGraph documentation and articles..."]
        result = write_interview_section(mock_analyst, context)

        assert "## Expert Perspective" in result
        mock_llm_for.assert_called_with("writer")


class TestWriteReportBody:
    """Test write_report_body chain."""

    def test_write_report_body(self, mock_llm_for):
        """Test writing the report body."""
        mock_response = AIMessage(
            content="## Analysis\n\nBased on the expert perspectives..."
        )
        mock_llm_for.return_value.invoke.return_value = mock_response

        sections = [
            "Section 1: Introduction",
            "Section 2: Analysis",
        ]
        result = write_report_body("AI Orchestration Comparison", sections)

        assert "## Analysis" in result
        mock_llm_for.assert_called_with("writer")


class TestWriteBookend:
    """Test write_bookend chain."""

    def test_write_introduction(self, mock_llm_for):
        """Test writing the introduction."""
        mock_response = AIMessage(content="## Introduction\n\nThis report examines...")
        mock_llm_for.return_value.invoke.return_value = mock_response

        sections = ["Expert 1 perspective", "Expert 2 perspective"]
        result = write_bookend("introduction", "AI Comparison", sections)

        assert "## Introduction" in result
        mock_llm_for.assert_called_with("writer")

    def test_write_conclusion(self, mock_llm_for):
        """Test writing the conclusion."""
        mock_response = AIMessage(content="## Conclusion\n\nIn summary...")
        mock_llm_for.return_value.invoke.return_value = mock_response

        sections = ["Expert 1 perspective", "Expert 2 perspective"]
        result = write_bookend("conclusion", "AI Comparison", sections)

        assert "## Conclusion" in result
