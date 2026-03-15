"""Unit tests for src.postprocess modules."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# confidence_gate
# ---------------------------------------------------------------------------

class TestFilterLowConfidence:
    def _make_outputs(self, confidences: list[float]) -> list[dict]:
        return [
            {
                "line_id": f"line_{i}",
                "top1_text": f"text_{i}",
                "sequence_confidence": conf,
            }
            for i, conf in enumerate(confidences)
        ]

    def test_all_above_threshold(self):
        from src.postprocess.confidence_gate import filter_low_confidence

        outputs = self._make_outputs([0.9, 0.8, 0.95])
        flagged, accepted = filter_low_confidence(outputs, threshold=0.7)
        assert len(flagged) == 0
        assert len(accepted) == 3

    def test_all_below_threshold(self):
        from src.postprocess.confidence_gate import filter_low_confidence

        outputs = self._make_outputs([0.3, 0.4, 0.5])
        flagged, accepted = filter_low_confidence(outputs, threshold=0.7)
        assert len(flagged) == 3
        assert len(accepted) == 0

    def test_mixed(self):
        from src.postprocess.confidence_gate import filter_low_confidence

        outputs = self._make_outputs([0.9, 0.3, 0.8, 0.1])
        flagged, accepted = filter_low_confidence(outputs, threshold=0.7)
        assert len(flagged) == 2
        assert len(accepted) == 2

    def test_max_flags_cap(self):
        from src.postprocess.confidence_gate import filter_low_confidence

        outputs = self._make_outputs([0.1] * 10)
        flagged, accepted = filter_low_confidence(outputs, threshold=0.7, max_flags=3)
        assert len(flagged) == 3
        assert len(accepted) == 7


# ---------------------------------------------------------------------------
# llm_correction
# ---------------------------------------------------------------------------

class TestRequestCorrection:
    def test_returns_audit_entry(self):
        from src.postprocess.llm_correction import request_correction, MockLLMClient

        client = MockLLMClient()
        entry = request_correction(
            line_id="src_001_l0",
            ocr_text="eñ la çibdad",
            nbest=["eñ la çibdad", "en la ciudad"],
            client=client,
        )
        assert entry["line_id"] == "src_001_l0"
        assert entry["input_text"] == "eñ la çibdad"
        assert "candidate_text" in entry

    def test_decision_is_pending(self):
        from src.postprocess.llm_correction import request_correction, MockLLMClient

        entry = request_correction(
            line_id="l0", ocr_text="foo", nbest=["foo"], client=MockLLMClient()
        )
        assert entry["decision"] == "pending"

    def test_audit_has_required_keys(self):
        from src.postprocess.llm_correction import request_correction, MockLLMClient

        entry = request_correction(
            line_id="l0", ocr_text="foo", nbest=["foo"], client=MockLLMClient()
        )
        for key in ("line_id", "input_text", "candidate_text", "decision", "decision_reason"):
            assert key in entry

    def test_build_prompt_contains_ocr_text(self):
        from src.postprocess.llm_correction import build_prompt

        prompt = build_prompt("eñ la çibdad", ["eñ la çibdad", "en la ciudad"])
        assert "eñ la çibdad" in prompt

    def test_failed_llm_falls_back_to_original(self):
        from src.postprocess.llm_correction import request_correction

        class FailingClient:
            def complete(self, prompt: str) -> str:
                raise RuntimeError("API error")

        entry = request_correction(
            line_id="l0", ocr_text="original", nbest=["original"], client=FailingClient()
        )
        assert entry["candidate_text"] == "original"


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

class TestValidateCorrection:
    def _make_entry(self, original: str, candidate: str) -> dict:
        return {
            "line_id": "l0",
            "input_text": original,
            "candidate_text": candidate,
            "decision": "pending",
            "decision_reason": "",
        }

    def test_valid_correction_accepted(self):
        from src.postprocess.validator import validate_correction

        entry = self._make_entry("holla", "hola")
        result = validate_correction(entry)
        assert result["decision"] == "accepted"

    def test_empty_candidate_rejected(self):
        from src.postprocess.validator import validate_correction

        entry = self._make_entry("hola", "")
        result = validate_correction(entry)
        assert result["decision"] == "rejected"
        assert "empty" in result["decision_reason"]

    def test_whitespace_only_rejected(self):
        from src.postprocess.validator import validate_correction

        entry = self._make_entry("hola", "   ")
        result = validate_correction(entry)
        assert result["decision"] == "rejected"

    def test_length_ratio_exceeded_rejected(self):
        from src.postprocess.validator import validate_correction

        entry = self._make_entry("hi", "a" * 100)
        result = validate_correction(entry, max_length_ratio=2.0)
        assert result["decision"] == "rejected"
        assert "length ratio" in result["decision_reason"]

    def test_illegal_chars_rejected(self):
        from src.postprocess.validator import validate_correction

        allowed = set("abcdefghijklmnopqrstuvwxyz ")
        entry = self._make_entry("hello", "hel1o")
        result = validate_correction(entry, allowed_chars=allowed)
        assert result["decision"] == "rejected"
        assert "disallowed" in result["decision_reason"]

    def test_lexicon_protection_accepts_when_coverage_maintained(self):
        from src.decode.lexicon import Lexicon
        from src.postprocess.validator import validate_correction

        lex = Lexicon.from_iterable(["dios"])
        entry = self._make_entry("dios mio", "dios mío")  # keeps 'dios'
        result = validate_correction(entry, lexicon=lex)
        assert result["decision"] == "accepted"

    def test_lexicon_protection_rejects_when_all_lex_words_removed(self):
        from src.decode.lexicon import Lexicon
        from src.postprocess.validator import validate_correction

        lex = Lexicon.from_iterable(["dios"])
        entry = self._make_entry("dios mio", "xyz abc")  # removes 'dios'
        result = validate_correction(entry, lexicon=lex)
        assert result["decision"] == "rejected"
        assert "lexicon" in result["decision_reason"]

    def test_does_not_mutate_input(self):
        from src.postprocess.validator import validate_correction

        entry = self._make_entry("hola", "hola mundo")
        original_entry = dict(entry)
        validate_correction(entry)
        assert entry == original_entry
