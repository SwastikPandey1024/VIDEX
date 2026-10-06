"""Unit tests for VIDEX Claim models and ClaimValidator anti-hallucination rules."""

from __future__ import annotations

import pytest

from videx.agent.claims import Claim, ClaimSet, ClaimValidator
from videx.agent.types import ClaimStatus


def test_claim_creation_and_with_status():
    claim = Claim.create(
        video_id="vid-1",
        claim_text="Person entered scene at 1.0s",
        confidence=0.95,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-1"],
        event_ids=["event-1"],
        timestamps=(1.0, 2.5),
        participant_ids=["track-1"],
    )
    assert claim.video_id == "vid-1"
    assert claim.status == ClaimStatus.SUPPORTED
    assert claim.evidence_ids == ("ev-1",)
    assert claim.timestamps == (1.0, 2.5)

    updated = claim.with_status(ClaimStatus.REJECTED)
    assert updated.status == ClaimStatus.REJECTED
    assert updated.claim_id == claim.claim_id
    assert claim.status == ClaimStatus.SUPPORTED


def test_claim_set_properties():
    c1 = Claim.create(
        video_id="vid-1",
        claim_text="Observation 1",
        confidence=0.9,
        epistemic_status="deterministically observed",
        status=ClaimStatus.SUPPORTED,
        evidence_ids=["ev-1"],
    )
    c2 = Claim.create(
        video_id="vid-1",
        claim_text="Observation 2",
        confidence=0.8,
        epistemic_status="heuristically associated",
        status=ClaimStatus.SUPPORTED,
        evidence_ids=["ev-2"],
    )
    c_set = ClaimSet(claims=[c1, c2])
    assert c_set.is_fully_supported
    assert not c_set.has_insufficient_evidence
    assert pytest.approx(c_set.overall_confidence) == 0.85

    c_insufficient = Claim.create(
        video_id="vid-1",
        claim_text="Unknown observation",
        confidence=0.0,
        epistemic_status="insufficient evidence",
        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
    )
    c_set_mixed = ClaimSet(claims=[c1, c_insufficient])
    assert not c_set_mixed.is_fully_supported
    assert c_set_mixed.has_insufficient_evidence


def test_claim_validator_valid_claim():
    validator = ClaimValidator()
    claim = Claim.create(
        video_id="vid-1",
        claim_text="Valid claim",
        confidence=0.92,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-1"],
        event_ids=["evnt-1"],
        timestamps=(1.0, 3.0),
        participant_ids=["trk-1"],
    )
    report = validator.validate(
        claims=[claim],
        expected_video_id="vid-1",
        known_evidence_ids={"ev-1"},
        known_event_ids={"evnt-1"},
        known_participant_ids={"trk-1"},
    )
    assert report.is_valid
    assert len(report.validated_claims) == 1
    assert len(report.rejected_claims) == 0


def test_claim_validator_video_isolation_mismatch():
    validator = ClaimValidator()
    claim = Claim.create(
        video_id="vid-2",
        claim_text="Cross-video claim",
        confidence=0.9,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-1"],
    )
    report = validator.validate(
        claims=[claim],
        expected_video_id="vid-1",
    )
    assert not report.is_valid
    assert len(report.rejected_claims) == 1
    assert "video_id" in report.errors[0]


def test_claim_validator_supported_claim_missing_evidence():
    validator = ClaimValidator()
    claim = Claim.create(
        video_id="vid-1",
        claim_text="Unsupported assertion",
        confidence=0.95,
        epistemic_status="deterministically observed",
        evidence_ids=[],
    )
    report = validator.validate(
        claims=[claim],
        expected_video_id="vid-1",
    )
    assert not report.is_valid
    assert len(report.rejected_claims) == 1
    assert "cites no evidence_ids" in report.errors[0]


def test_claim_validator_invalid_timestamps():
    validator = ClaimValidator()
    # Negative start
    c_neg = Claim.create(
        video_id="vid-1",
        claim_text="Negative time",
        confidence=0.8,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-1"],
        timestamps=(-1.0, 2.0),
    )
    report_neg = validator.validate(claims=[c_neg], expected_video_id="vid-1")
    assert not report_neg.is_valid

    # Inverted start > end
    c_inv = Claim.create(
        video_id="vid-1",
        claim_text="Inverted time",
        confidence=0.8,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-1"],
        timestamps=(5.0, 2.0),
    )
    report_inv = validator.validate(claims=[c_inv], expected_video_id="vid-1")
    assert not report_inv.is_valid


def test_claim_validator_unknown_identifiers():
    validator = ClaimValidator()
    claim = Claim.create(
        video_id="vid-1",
        claim_text="Hallucinated IDs",
        confidence=0.8,
        epistemic_status="deterministically observed",
        evidence_ids=["ev-fake"],
        event_ids=["event-fake"],
        participant_ids=["track-fake"],
    )
    report = validator.validate(
        claims=[claim],
        expected_video_id="vid-1",
        known_evidence_ids={"ev-real"},
        known_event_ids={"event-real"},
        known_participant_ids={"track-real"},
    )
    assert not report.is_valid
    assert len(report.errors) == 3
