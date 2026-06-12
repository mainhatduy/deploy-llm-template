from typing import List
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    query_id: str = Field(..., description="Unique query identifier", min_length=1)
    type: str = Field(..., description="Query type: 'type1' or 'type2'")
    query: str = Field(..., description="The question (Type 1) or full problem statement (Type 2)", min_length=1)
    premises: List[str] = Field(default_factory=list, description="Natural-language premises, 0-indexed")
    options: List[str] = Field(default_factory=list, description="Choice set for multiple-choice questions")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ("type1", "type2"):
            raise ValueError("type must be either 'type1' or 'type2'")
        return v

class PredictResponse(BaseModel):
    query_id: str = Field(..., description="Unique query identifier corresponding to the request")
    answer: str = Field(..., description="The predicted answer (exactly one of the options for choice questions, or free-form text/number)")
    premises_used: List[int] = Field(default_factory=list, description="Indices of the premises used to answer the query")
