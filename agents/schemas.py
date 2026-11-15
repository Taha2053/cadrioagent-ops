"""Research report schema and deterministic evidence/safety checks."""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = "Educational research only. Not a diagnosis or treatment recommendation. Seek qualified clinical assessment."


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=25, max_length=700)


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=10, max_length=1600)
    limitations: str = Field(min_length=20, max_length=1200)
    citations: list[Citation] = Field(min_length=1, max_length=4)
    next_step: Literal["Review this recording and evidence with a qualified clinician."]


def audit(report: Report, sources: list[dict]) -> list[str]:
    allowed = {s["id"]: " ".join(s["text"].split()) for s in sources}
    errors = []
    for citation in report.citations:
        if citation.source_id not in allowed:
            errors.append("Unknown citation ID")
        elif " ".join(citation.quote.split()) not in allowed[citation.source_id]:
            errors.append("Citation quote does not match retrieved evidence")
    text = report.summary + " " + report.limitations
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|tablets?|doses?)\b|\b(?:you have|diagnosis is|definitely|certainly|prescribe|start taking|stop taking|ignore previous|system prompt)\b", text, re.I):
        errors.append("Unsafe or instruction-like report content")
    return errors
