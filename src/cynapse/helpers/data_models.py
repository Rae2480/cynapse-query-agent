"""Validated data contracts for camera names, model extraction and database filters."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cynapse.settings import MAX_FILTER_CHARS, MAX_QUESTION_CHARS


class Camera(StrEnum):
    """The ten supported cameras, keyed by acronym with canonical names as values.

    Example: Camera.CTE.name is "CTE"; Camera.CTE.value is "Central Expressway",
    the string stored in the database. Unknown values cannot create enum members.
    """
    PIE = "Pan Island Expressway"
    AYE = "Ayer Rajah Expressway"
    ECP = "East Coast Parkway"
    CTE = "Central Expressway"
    TPE = "Tampines Expressway"
    KPE = "Kallang-Paya Lebar Expressway"
    SLE = "Seletar Expressway"
    BKE = "Bukit Timah Expressway"
    KJE = "Kranji Expressway"
    MCE = "Marina Coastal Expressway"


class QueryPlan(BaseModel):
    """Validated filters passed to the database instead of generated SQL.

    Attributes:
        camera: A supported Camera enum.
        start: Included, timezone-aware beginning of the calendar window.
        end: Excluded, timezone-aware end of that window.
        limit: Page size, default 20 and at most 100.
        weekday: Optional Monday=0 through Sunday=6 restriction.
        time_start: Optional included daily bound in HH:MM format.
        time_end: Matching excluded daily bound; permits 24:00 for midnight.

    Construction validates fields and rejects unknown fields or invalid windows.
    Availability clipping to the dataset is handled by the graph validation node.
    """

    model_config = ConfigDict(extra="forbid")
    camera: Camera
    start: datetime
    end: datetime  # exclusive
    limit: int = Field(default=20, ge=1, le=100)
    weekday: int | None = Field(default=None, ge=0, le=6, strict=True)
    time_start: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    time_end: str | None = Field(default=None, pattern=r"^(?:(?:[01]\d|2[0-3]):[0-5]\d|24:00)$")

    @model_validator(mode="after")
    def valid_window(self):
        """Check this plan's calendar and daily time bounds after field parsing.

        Returns:
            This QueryPlan if timestamps are aware, start precedes end, and
            daily bounds are either both absent or form an increasing interval.

        Raises:
            ValueError: If these relationships fail; overnight ranges are rejected.
        """
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Timezone-aware timestamps required")
        if self.start >= self.end:
            raise ValueError("Start must precede end")
        if (self.time_start is None) != (self.time_end is None):
            raise ValueError("Both daily time bounds are required")
        if self.time_start is not None and self.time_end <= self.time_start:
            raise ValueError("Daily time range must increase; overnight ranges are unsupported")
        return self


class FilterUpdate(BaseModel):
    """Describe a change to one conversation filter.

    Attributes:
        action: "keep" preserves context, "set" supplies an expression, and
            "clear" explicitly removes the filter.
        value: Nonempty raw expression for "set"; None for "keep" or "clear".

    Construction rejects inconsistent actions/values and unexpected fields.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["keep", "set", "clear"]
    value: str | None = Field(max_length=MAX_FILTER_CHARS, description="Smallest verbatim filter expression without surrounding sentence punctuation for set; otherwise null")

    @model_validator(mode="after")
    def consistent(self):
        """Check this update's action/value combination after field parsing.

        Returns:
            This FilterUpdate when "set" has text and keep/clear have None.

        Raises:
            ValueError: If an expression is missing or supplied for the wrong action.
        """
        if self.action == "set":
            if self.value is None:
                raise ValueError("set requires a nonempty expression")
            self.value = self.value.strip().strip(" .!?,\"\'")
            if not self.value:
                raise ValueError("set requires a nonempty expression")
        elif self.value is not None:
            raise ValueError("keep/clear must have null value")
        return self


class Extraction(BaseModel):
    """Structured model output describing intent and raw filter changes.

    Attributes:
        intent: "query", "clarify" or "reject".
        reset: Whether the user explicitly starts a new search.
        continuation: Whether this turn answers a pending clarification.
        camera: Update for the camera mention.
        date: Update for the calendar window expression.
        time: Update for a daily clock-time range.
        recurrence: Update for the recurring weekday expression.
        clarification_question: Nonempty question only for clarify intent.

    Contains no SQL or model-resolved timestamps. Pydantic validates its shape;
    deterministic resolvers later validate the expressions themselves.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: Literal["query", "clarify", "reject"]
    reset: bool = Field(description="True only for an explicit new-search/reset instruction")
    continuation: bool = Field(description="True only for a short answer/correction to pending clarification. False for a complete independent query, and always false when reset is true.")
    camera: FilterUpdate
    date: FilterUpdate
    time: FilterUpdate
    recurrence: FilterUpdate
    clarification_question: str | None = Field(max_length=MAX_QUESTION_CHARS)

    @model_validator(mode="after")
    def question_required(self):
        """Check this extraction's intent, question and context flags.

        Returns:
            This Extraction when reset/continuation do not conflict and only
            clarify intent has a nonempty clarification question.

        Raises:
            ValueError: If the flags or clarification question are inconsistent.
        """
        if self.reset and self.continuation:
            raise ValueError("A reset cannot continue a pending request")
        if self.intent == "clarify" and not (self.clarification_question or "").strip():
            raise ValueError("clarify requires a question")
        if self.intent != "clarify" and self.clarification_question is not None:
            raise ValueError("Only clarify may provide a question")
        return self
