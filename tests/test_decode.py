"""Decoder regression tests.

All tests operate on fixed, deterministic logit arrays so they can be replayed
without a trained model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Fixed vocabulary used across all tests.
VOCAB = list(" abcdefghijklmnopqrstuvwxyz")  # len=27, blank=index 0 (space)
# We'll use blank_idx=0 and map vocab[0]=' ' to the blank concept for simplicity,
# but the blank token is separate; let's define a clean vocab.
VOCAB_BLANK = ["<blank>"] + list(" abcdefghijklmnopqrstuvwxyz")  # len=28, blank=0
BLANK_IDX = 0


def _one_hot_logits(indices: list[int], num_classes: int = 28) -> np.ndarray:
    """Build a (T, C) log-probability array that decodes deterministically."""
    T = len(indices)
    logits = np.full((T, num_classes), -10.0)
    for t, idx in enumerate(indices):
        logits[t, idx] = 0.0  # log(1) = 0
    return logits


# ---------------------------------------------------------------------------
# greedy_decoder
# ---------------------------------------------------------------------------

class TestGreedyDecode:
    def test_decodes_simple_word(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        # Spell out "hola": h=9, o=16, l=13, a=2 (1-indexed after blank)
        # VOCAB_BLANK[0]=blank, [1]=space, [2]=a, [3]=b, ..., [9]=h, [13]=l, [16]=o
        h = VOCAB_BLANK.index("h")
        o = VOCAB_BLANK.index("o")
        l = VOCAB_BLANK.index("l")
        a = VOCAB_BLANK.index("a")
        indices = [h, h, BLANK_IDX, o, l, l, BLANK_IDX, a]
        logits = _one_hot_logits(indices)
        result = greedy_decode(logits, VOCAB_BLANK, blank_idx=BLANK_IDX)
        assert result["text"] == "hola"

    def test_collapses_repeats(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        # "aa" should collapse to "a".
        a = VOCAB_BLANK.index("a")
        logits = _one_hot_logits([a, a, a])
        result = greedy_decode(logits, VOCAB_BLANK, blank_idx=BLANK_IDX)
        assert result["text"] == "a"

    def test_blank_separates_repeats(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        # "aa" with blank → "aa".
        a = VOCAB_BLANK.index("a")
        logits = _one_hot_logits([a, BLANK_IDX, a])
        result = greedy_decode(logits, VOCAB_BLANK, blank_idx=BLANK_IDX)
        assert result["text"] == "aa"

    def test_empty_result_for_all_blanks(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        logits = _one_hot_logits([BLANK_IDX, BLANK_IDX, BLANK_IDX])
        result = greedy_decode(logits, VOCAB_BLANK, blank_idx=BLANK_IDX)
        assert result["text"] == ""

    def test_wrong_shape_raises(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        logits = np.zeros((5, 10, 3))
        with pytest.raises(ValueError, match="2-D"):
            greedy_decode(logits, VOCAB_BLANK)

    def test_vocab_length_mismatch_raises(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        logits = np.zeros((5, 28))
        with pytest.raises(ValueError, match="vocab length"):
            greedy_decode(logits, VOCAB_BLANK[:10])

    def test_deterministic(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        logits = _one_hot_logits([2, BLANK_IDX, 3, 4])
        r1 = greedy_decode(logits, VOCAB_BLANK)
        r2 = greedy_decode(logits, VOCAB_BLANK)
        assert r1["text"] == r2["text"]

    def test_returns_raw_indices(self):
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        indices = [2, BLANK_IDX, 3]
        logits = _one_hot_logits(indices)
        result = greedy_decode(logits, VOCAB_BLANK)
        assert result["raw_indices"] == indices


# ---------------------------------------------------------------------------
# beam_search
# ---------------------------------------------------------------------------

class TestBeamDecode:
    def test_returns_list(self):
        from autotrandhd.core.decoding.beam_search import beam_decode

        logits = _one_hot_logits([2, BLANK_IDX, 3])
        results = beam_decode(logits, VOCAB_BLANK, beam_width=5)
        assert isinstance(results, list)

    def test_top1_matches_greedy(self):
        from autotrandhd.core.decoding.beam_search import beam_decode
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        indices = [VOCAB_BLANK.index("d"), BLANK_IDX, VOCAB_BLANK.index("e")]
        logits = _one_hot_logits(indices)
        beam_top = beam_decode(logits, VOCAB_BLANK, beam_width=5)
        greedy_r = greedy_decode(logits, VOCAB_BLANK)
        # For deterministic logits, top beam == greedy result.
        assert beam_top[0]["text"] == greedy_r["text"]

    def test_n_best_bounded(self):
        from autotrandhd.core.decoding.beam_search import beam_decode

        logits = _one_hot_logits([2, 3, 4])
        results = beam_decode(logits, VOCAB_BLANK, beam_width=3, max_candidates=2)
        assert len(results) <= 2

    def test_scores_descending(self):
        from autotrandhd.core.decoding.beam_search import beam_decode

        rng = np.random.default_rng(0)
        logits = rng.random((10, len(VOCAB_BLANK)))
        logits = np.log(logits / logits.sum(axis=1, keepdims=True) + 1e-10)
        results = beam_decode(logits, VOCAB_BLANK, beam_width=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_wrong_shape_raises(self):
        from autotrandhd.core.decoding.beam_search import beam_decode

        with pytest.raises(ValueError, match="2-D"):
            beam_decode(np.zeros((5, 10, 3)), VOCAB_BLANK)


# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------

class TestLexicon:
    def test_contains(self, tmp_path):
        from autotrandhd.core.decoding.lexicon import Lexicon

        lex_file = tmp_path / "lex.txt"
        lex_file.write_text("dios\nrey 2.0\n# comment\n")
        lex = Lexicon.from_file(lex_file)
        assert lex.contains("dios")
        assert lex.contains("rey")
        assert not lex.contains("foo")

    def test_score(self, tmp_path):
        from autotrandhd.core.decoding.lexicon import Lexicon

        lex_file = tmp_path / "lex.txt"
        lex_file.write_text("rey 3.5\n")
        lex = Lexicon.from_file(lex_file)
        assert lex.score("rey") == pytest.approx(3.5)
        assert lex.score("unknown") == 0.0

    def test_from_iterable(self):
        from autotrandhd.core.decoding.lexicon import Lexicon

        lex = Lexicon.from_iterable(["a", "b", "c"])
        assert lex.contains("a")
        assert len(lex) == 3

    def test_save_reload(self, tmp_path):
        from autotrandhd.core.decoding.lexicon import Lexicon

        lex = Lexicon.from_iterable(["dios", "rey"], default_score=1.0)
        out_path = tmp_path / "saved.txt"
        lex.save(out_path)
        lex2 = Lexicon.from_file(out_path)
        assert lex2.contains("dios")
        assert lex2.contains("rey")

    def test_missing_file_raises(self, tmp_path):
        from autotrandhd.core.decoding.lexicon import Lexicon

        with pytest.raises(FileNotFoundError):
            Lexicon.from_file(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def _make_peaked_logits(self, indices: list[int], C: int = 28) -> np.ndarray:
        """Near-deterministic logits for testing."""
        logits = np.full((len(indices), C), -8.0)
        for t, idx in enumerate(indices):
            logits[t, idx] = 0.0
        return logits

    def test_token_confidence_range(self):
        from autotrandhd.core.decoding.confidence import token_confidence
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        indices = [2, BLANK_IDX, 3]
        logits = self._make_peaked_logits(indices)
        result = greedy_decode(logits, VOCAB_BLANK)
        confs = token_confidence(logits, result["raw_indices"])
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_sequence_confidence_range(self):
        from autotrandhd.core.decoding.confidence import sequence_confidence

        assert 0.0 <= sequence_confidence([0.8, 0.9, 0.7]) <= 1.0

    def test_empty_sequence_confidence(self):
        from autotrandhd.core.decoding.confidence import sequence_confidence

        assert sequence_confidence([]) == 0.0

    def test_high_conf_peaked_logits(self):
        from autotrandhd.core.decoding.confidence import token_confidence, sequence_confidence
        from autotrandhd.core.decoding.greedy_decoder import greedy_decode

        indices = [2, BLANK_IDX, 3]
        logits = self._make_peaked_logits(indices)
        result = greedy_decode(logits, VOCAB_BLANK)
        confs = token_confidence(logits, result["raw_indices"])
        seq = sequence_confidence(confs)
        assert seq > 0.5
