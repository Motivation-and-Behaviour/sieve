from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ScreeningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vote: Literal["include", "exclude"] = Field(
        description="Final decision. Choose exactly one."
    )
    matched_inclusion: list[int] | None
    failed_inclusion: list[int] | None
    triggered_exclusion: list[int] | None
    exclusion_reasons: list[str] | None
    rationale: str
