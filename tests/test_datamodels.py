"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from sieve.datamodels import ScreeningDecision


class TestScreeningDecision:
    def test_include_vote(self):
        decision = ScreeningDecision(
            vote="include",
            matched_inclusion=[1, 2, 3],
            failed_inclusion=None,
            triggered_exclusion=None,
            exclusion_reasons=None,
            rationale="All criteria met.",
        )
        assert decision.vote == "include"
        assert decision.matched_inclusion == [1, 2, 3]
        assert decision.rationale == "All criteria met."

    def test_exclude_vote(self):
        decision = ScreeningDecision(
            vote="exclude",
            matched_inclusion=None,
            failed_inclusion=[1],
            triggered_exclusion=[2],
            exclusion_reasons=["No screen time measure"],
            rationale="Failed inclusion criterion 1.",
        )
        assert decision.vote == "exclude"
        assert decision.failed_inclusion == [1]
        assert decision.triggered_exclusion == [2]

    def test_invalid_vote_raises_error(self):
        with pytest.raises(ValidationError):
            ScreeningDecision(vote="maybe", rationale="Unsure")

    def test_rationale_required(self):
        with pytest.raises(ValidationError):
            ScreeningDecision(vote="include")

    def test_json_validation(self):
        json_str = '{"vote": "include", "matched_inclusion": [1, 2], "failed_inclusion": null, "triggered_exclusion": null, "exclusion_reasons": null, "rationale": "All good"}'  # noqa: E501
        decision = ScreeningDecision.model_validate_json(json_str)
        assert decision.vote == "include"
        assert decision.rationale == "All good"

    def test_model_dump(self):
        decision = ScreeningDecision(
            vote="exclude",
            matched_inclusion=None,
            failed_inclusion=None,
            triggered_exclusion=[1],
            exclusion_reasons=None,
            rationale="Exclusion triggered",
        )
        dumped = decision.model_dump()
        assert dumped["vote"] == "exclude"
        assert dumped["triggered_exclusion"] == [1]
        assert dumped["rationale"] == "Exclusion triggered"
