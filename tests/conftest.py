"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def mock_llm():
    """Mock ChatAnthropic instance with structured output support."""
    mock = MagicMock()
    # Setup the invoke method
    mock.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    # Setup with_structured_output to return a chain that has invoke
    structured_chain = MagicMock()
    structured_chain.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    mock.with_structured_output = MagicMock(return_value=structured_chain)
    return mock


@pytest.fixture
def mock_llm_for(mock_llm):
    """Mock the llm_for function to return the mock LLM."""
    # Patch at the chains module level where it's used
    with patch("research_pipeline.chains.llm_for", return_value=mock_llm) as mock_fn:
        yield mock_fn


@pytest.fixture
def mock_analyst():
    """Create a mock Analyst instance."""
    from research_pipeline.schemas import Analyst
    
    return Analyst(
        name="Dr. Jane Smith",
        role="Research Analyst",
        affiliation="Tech Institute",
        description="Focuses on AI and machine learning trends. Interested in practical applications and industry adoption patterns.",
    )


@pytest.fixture
def mock_perspectives():
    """Mock Perspectives response."""
    from research_pipeline.schemas import Analyst, Perspectives
    
    analysts = [
        Analyst(
            name="Dr. Alice Johnson",
            role="ML Engineer",
            affiliation="AI Corp",
            description="Focuses on implementation and performance. Concerned about scalability and production readiness.",
        ),
        Analyst(
            name="Dr. Bob Chen",
            role="Researcher",
            affiliation="Tech University",
            description="Focuses on theoretical foundations and emerging research. Interested in novel approaches.",
        ),
    ]
    return Perspectives(analysts=analysts)


@pytest.fixture
def mock_search_query():
    """Mock SearchQuery response."""
    from research_pipeline.schemas import SearchQuery
    
    return SearchQuery(search_query="LangGraph vs Temporal production deployment")
