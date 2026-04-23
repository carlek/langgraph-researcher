"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from research_pipeline.schemas import Analyst, SearchQuery, Perspectives


class TestAnalyst:
    """Test Analyst schema."""

    def test_analyst_creation(self):
        """Test creating a valid Analyst."""
        analyst = Analyst(
            name="Dr. Jane Smith",
            role="Research Analyst",
            affiliation="Tech Institute",
            description="Focuses on AI trends and market analysis. Experienced in technology adoption.",
        )
        assert analyst.name == "Dr. Jane Smith"
        assert analyst.role == "Research Analyst"
        assert analyst.affiliation == "Tech Institute"

    def test_analyst_persona_property(self):
        """Test the persona property formats correctly."""
        analyst = Analyst(
            name="Dr. Jane Smith",
            role="Research Analyst",
            affiliation="Tech Institute",
            description="Focuses on AI trends and market analysis consistently.",
        )
        persona = analyst.persona
        assert "Name: Dr. Jane Smith" in persona
        assert "Role: Research Analyst" in persona
        assert "Affiliation: Tech Institute" in persona

    def test_analyst_strips_whitespace(self):
        """Test that fields are stripped of whitespace."""
        analyst = Analyst(
            name="  Dr. Jane Smith  ",
            role="  Research Analyst  ",
            affiliation="  Tech Institute  ",
            description="  Focuses on trends.  ",
        )
        assert analyst.name == "Dr. Jane Smith"
        assert analyst.role == "Research Analyst"
        assert analyst.affiliation == "Tech Institute"
        assert analyst.description == "Focuses on trends."

    def test_analyst_frozen(self):
        """Test that Analyst instances are frozen."""
        analyst = Analyst(
            name="Dr. Jane Smith",
            role="Research Analyst",
            affiliation="Tech Institute",
            description="Focuses on AI trends with extensive experience in market analysis.",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            analyst.name = "Different Name"

    def test_analyst_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            Analyst(
                name="",
                role="Research Analyst",
                affiliation="Tech Institute",
                description="Focuses on trends.",
            )

    def test_analyst_description_min_length(self):
        """Test that description has minimum length."""
        with pytest.raises(ValidationError):
            Analyst(
                name="Dr. Jane Smith",
                role="Research Analyst",
                affiliation="Tech Institute",
                description="Short",
            )


class TestSearchQuery:
    """Test SearchQuery schema."""

    def test_search_query_creation(self):
        """Test creating a valid SearchQuery."""
        query = SearchQuery(search_query="LangGraph vs Temporal comparison")
        assert query.search_query == "LangGraph vs Temporal comparison"

    def test_search_query_empty_allowed(self):
        """Test that SearchQuery allows empty strings (no validation constraints)."""
        query = SearchQuery(search_query="")
        assert query.search_query == ""


class TestPerspectives:
    """Test Perspectives schema."""

    def test_perspectives_creation(self, mock_perspectives):
        """Test creating Perspectives with analysts."""
        assert len(mock_perspectives.analysts) == 2
        assert mock_perspectives.analysts[0].name == "Dr. Alice Johnson"

    def test_perspectives_empty(self):
        """Test creating empty Perspectives."""
        from research_pipeline.schemas import Perspectives
        
        perspectives = Perspectives(analysts=[])
        assert perspectives.analysts == []
