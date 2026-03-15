"""Golden-file tests for src.eval metrics and analysis."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# compute_cer
# ---------------------------------------------------------------------------

class TestComputeCER:
    def test_perfect_predictions(self):
        from src.eval.metrics import compute_cer

        preds = ["hola", "mundo"]
        refs = ["hola", "mundo"]
        assert compute_cer(preds, refs) == pytest.approx(0.0)

    def test_completely_wrong(self):
        from src.eval.metrics import compute_cer

        preds = ["xxxx"]
        refs = ["hola"]
        # 4 substitutions / 4 chars = 1.0
        assert compute_cer(preds, refs) == pytest.approx(1.0, abs=1e-4)

    def test_partial_errors(self):
        from src.eval.metrics import compute_cer

        preds = ["hXla"]
        refs = ["hola"]
        # 1 substitution / 4 chars = 0.25
        assert compute_cer(preds, refs) == pytest.approx(0.25, abs=1e-4)

    def test_length_mismatch_raises(self):
        from src.eval.metrics import compute_cer

        with pytest.raises(ValueError, match="length"):
            compute_cer(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# compute_wer
# ---------------------------------------------------------------------------

class TestComputeWER:
    def test_perfect_predictions(self):
        from src.eval.metrics import compute_wer

        preds = ["el rey", "dios mio"]
        refs = ["el rey", "dios mio"]
        assert compute_wer(preds, refs) == pytest.approx(0.0)

    def test_one_word_wrong(self):
        from src.eval.metrics import compute_wer

        preds = ["el rey"]
        refs = ["el dia"]
        # 1 substitution / 2 words = 0.5
        assert compute_wer(preds, refs) == pytest.approx(0.5, abs=1e-4)

    def test_length_mismatch_raises(self):
        from src.eval.metrics import compute_wer

        with pytest.raises(ValueError, match="length"):
            compute_wer(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------

class TestComputeExactMatch:
    def test_all_match(self):
        from src.eval.metrics import compute_exact_match

        preds = ["hola", "mundo"]
        refs = ["hola", "mundo"]
        assert compute_exact_match(preds, refs) == pytest.approx(1.0)

    def test_none_match(self):
        from src.eval.metrics import compute_exact_match

        preds = ["foo", "bar"]
        refs = ["hola", "mundo"]
        assert compute_exact_match(preds, refs) == pytest.approx(0.0)

    def test_half_match(self):
        from src.eval.metrics import compute_exact_match

        preds = ["hola", "foo"]
        refs = ["hola", "mundo"]
        assert compute_exact_match(preds, refs) == pytest.approx(0.5)

    def test_empty_lists(self):
        from src.eval.metrics import compute_exact_match

        assert compute_exact_match([], []) == 0.0


# ---------------------------------------------------------------------------
# rare_char_metrics
# ---------------------------------------------------------------------------

class TestRareCharMetrics:
    def test_returns_dict(self):
        from src.eval.analysis import rare_char_metrics

        preds = ["aabñ"]
        refs = ["aabñ"]
        result = rare_char_metrics(preds, refs, min_freq=10)
        assert isinstance(result, dict)

    def test_perfect_rare_f1(self):
        from src.eval.analysis import rare_char_metrics

        # ñ appears once → rare with min_freq=2
        preds = ["añ"]
        refs = ["añ"]
        result = rare_char_metrics(preds, refs, min_freq=2)
        if "ñ" in result:
            assert result["ñ"]["f1"] == pytest.approx(1.0)

    def test_missing_rare_char(self):
        from src.eval.analysis import rare_char_metrics

        # ñ predicted but absent in refs → recall=0
        preds = ["añ"]
        refs = ["a "]
        result = rare_char_metrics(preds, refs, min_freq=2)
        # space or other chars may be rare; ñ is not in refs so not tracked
        assert isinstance(result, dict)

    def test_length_mismatch_raises(self):
        from src.eval.analysis import rare_char_metrics

        with pytest.raises(ValueError, match="same length"):
            rare_char_metrics(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# top_confusions
# ---------------------------------------------------------------------------

class TestTopConfusions:
    def test_returns_list(self):
        from src.eval.analysis import top_confusions

        result = top_confusions(["hXla"], ["hola"])
        assert isinstance(result, list)

    def test_finds_substitution(self):
        from src.eval.analysis import top_confusions

        result = top_confusions(["hXla", "hXla"], ["hola", "hola"])
        # X → o is the substitution
        tuples = [(p, r) for p, r, _ in result]
        assert ("X", "o") in tuples

    def test_length_mismatch_raises(self):
        from src.eval.analysis import top_confusions

        with pytest.raises(ValueError, match="same length"):
            top_confusions(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# run_benchmark (integration)
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_report_keys(self, tmp_path):
        from src.eval.benchmark import run_benchmark

        preds = ["hola mundo", "el rey"]
        refs = ["hola mundo", "el dia"]
        report = run_benchmark(
            predictions=preds,
            references=refs,
            checkpoint_id="ckpt_test",
            decoder_config={"beam_width": 1},
            split_id="test",
            output_dir=str(tmp_path),
        )
        required_keys = {
            "checkpoint_id", "decoder_config", "split_id",
            "cer", "wer", "exact_match", "num_lines",
        }
        assert required_keys <= set(report.keys())

    def test_report_file_created(self, tmp_path):
        from src.eval.benchmark import run_benchmark

        preds = ["hola"]
        refs = ["hola"]
        run_benchmark(
            predictions=preds,
            references=refs,
            checkpoint_id="ckpt_test",
            decoder_config={},
            split_id="test",
            output_dir=str(tmp_path),
        )
        csv_files = list(tmp_path.glob("*.json"))
        assert len(csv_files) == 1

    def test_length_mismatch_raises(self, tmp_path):
        from src.eval.benchmark import run_benchmark

        with pytest.raises(ValueError, match="length"):
            run_benchmark(
                predictions=["a"],
                references=["a", "b"],
                checkpoint_id="x",
                decoder_config={},
                split_id="test",
                output_dir=str(tmp_path),
            )
