
from pydantic import BaseModel, Field, field_validator


class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Search query for retrieval.")
    

class Analyst(BaseModel):
    """One analyst persona. Consumed by the LLM as a structured output
    target and later read by the interview subgraph as state['analyst']."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="Full name of the analyst that fits the persona.",
    )
    role: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Analyst's role in the context of the topic.",
    )
    affiliation: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Primary institutional or organizational affiliation.",
    )
    description: str = Field(
        ...,
        min_length=20,
        description="Focus areas, concerns, and motives. Two to four sentences.",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("name", "role", "affiliation", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Description: {self.description}\n"
        )


class Perspectives(BaseModel):
    """Structured-output wrapper for the analyst-generation LLM call."""

    analysts: list[Analyst] = Field(
        ...,
        description="Comprehensive list of analysts with their roles and affiliations.",
    )
