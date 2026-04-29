from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osc_grimoire.calibration import (
    CalibrationExample,
    diagnose_calibration_session,
    load_calibration_examples,
    write_calibration_metadata,
)
from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import add_voice_sample, create_spell, load_spellbook
from osc_grimoire.voice_recognizer import VoiceTemplateBackend


def test_diagnose_calibration_session_recommends_margin(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path / "data")
    book, spell = create_spell(book, "Alpha")
    sample_dir = book.data_dir / "samples" / f"spell_{spell.id}"
    sample_dir.mkdir(parents=True)
    for name in ("template_a.wav", "template_b.wav"):
        path = sample_dir / name
        path.touch()
        book = add_voice_sample(book, spell, path.relative_to(book.data_dir).as_posix())

    session_dir = tmp_path / "calibration" / "session"
    examples: list[CalibrationExample] = []
    positive = session_dir / "positives" / "alpha" / "attempt_001.wav"
    positive.parent.mkdir(parents=True)
    positive.touch()
    negative = session_dir / "negatives" / "negative_001.wav"
    negative.parent.mkdir(parents=True)
    negative.touch()
    examples.append(
        CalibrationExample(
            path=positive,
            kind="positive",
            expected_spell_id=spell.id,
            expected_spell_name=spell.name,
        )
    )
    examples.append(CalibrationExample(path=negative, kind="negative"))

    write_calibration_metadata(session_dir, examples)
    backend = VoiceTemplateBackend(
        name="fake",
        extract_path=lambda path, _config: np.array(
            [[0.0 if "negative" not in path.name else 10.0]], dtype=np.float32
        ),
        extract_array=lambda audio, _config, _sample_rate: audio,
        distance=lambda a, b: float(abs(a[0, 0] - b[0, 0])),
        aggregate=lambda distances: float(np.median(distances)),
    )

    report = diagnose_calibration_session(
        session_dir, book, VoiceRecognitionConfig(), backend
    )

    assert report.recommended_margin_min is not None
    recommended = next(
        r for r in report.sweep if r.margin_min == report.recommended_margin_min
    )
    assert recommended.negative_accepted == 0
    assert recommended.positive_wrong == 0


def test_diagnose_with_backend_stats_does_not_mutate_spellbook(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path / "data")
    book, spell = create_spell(book, "Alpha")
    sample_dir = book.data_dir / "samples" / f"spell_{spell.id}"
    sample_dir.mkdir(parents=True)
    for i in range(2):
        path = sample_dir / f"voice_{i + 1:03d}.wav"
        path.touch()
        book = add_voice_sample(book, spell, path.relative_to(book.data_dir).as_posix())
    spell = next(s for s in book.spells if s.id == spell.id)
    assert spell.intra_class_median is None

    session_dir = tmp_path / "calibration" / "session"
    attempt = session_dir / "positives" / "alpha" / "attempt_001.wav"
    attempt.parent.mkdir(parents=True)
    attempt.touch()
    write_calibration_metadata(
        session_dir,
        [
            CalibrationExample(
                path=attempt,
                kind="positive",
                expected_spell_id=spell.id,
                expected_spell_name=spell.name,
            )
        ],
    )

    backend = VoiceTemplateBackend(
        name="fake",
        extract_path=lambda path, _config: np.array(
            [[len(path.name)]], dtype=np.float32
        ),
        extract_array=lambda audio, _config, _sample_rate: audio,
        distance=lambda a, b: float(abs(a[0, 0] - b[0, 0])),
        aggregate=lambda distances: float(np.median(distances)),
    )

    report = diagnose_calibration_session(
        session_dir, book, VoiceRecognitionConfig(), backend
    )

    assert report.backend_name == "fake"
    assert report.examples
    assert next(s for s in book.spells if s.id == spell.id).intra_class_median is None


def test_diagnose_ignores_spells_not_labeled_in_session(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path / "data")
    book, expected = create_spell(book, "Expected")
    book, extra = create_spell(book, "Extra")
    for spell in (expected, extra):
        sample_dir = book.data_dir / "samples" / f"spell_{spell.id}"
        sample_dir.mkdir(parents=True)
        path = sample_dir / "voice_001.wav"
        path.touch()
        book = add_voice_sample(book, spell, path.relative_to(book.data_dir).as_posix())

    session_dir = tmp_path / "calibration" / "session"
    positive = session_dir / "positives" / "expected" / "attempt_001.wav"
    positive.parent.mkdir(parents=True)
    positive.touch()
    negative = session_dir / "negatives" / "negative_001.wav"
    negative.parent.mkdir(parents=True)
    negative.touch()
    write_calibration_metadata(
        session_dir,
        [
            CalibrationExample(
                path=positive,
                kind="positive",
                expected_spell_id=expected.id,
                expected_spell_name=expected.name,
            ),
            CalibrationExample(path=negative, kind="negative"),
        ],
    )

    backend = VoiceTemplateBackend(
        name="fake",
        extract_path=lambda path, _config: np.array(
            [[100.0 if "negative" in path.name else 0.0]], dtype=np.float32
        ),
        extract_array=lambda audio, _config, _sample_rate: audio,
        distance=lambda a, b: (
            0.0 if b[0, 0] == 100.0 else float(abs(a[0, 0] - b[0, 0]) + 10.0)
        ),
        aggregate=lambda distances: float(np.median(distances)),
    )

    report = diagnose_calibration_session(
        session_dir,
        book,
        VoiceRecognitionConfig(relative_margin_min=0.0),
        backend,
    )
    negative_report = next(d for d in report.examples if d.example.kind == "negative")

    assert negative_report.best_spell_id == expected.id
    assert negative_report.best_spell_name == expected.name
    assert negative_report.best_spell_id != extra.id


def test_calibration_metadata_round_trips_variant_fields(tmp_path: Path) -> None:
    session_dir = tmp_path / "calibration" / "session"
    path = session_dir / "positives" / "alpha" / "quiet" / "attempt_001.wav"
    path.parent.mkdir(parents=True)
    path.touch()

    write_calibration_metadata(
        session_dir,
        [
            CalibrationExample(
                path=path,
                kind="positive",
                expected_spell_id="spell-1",
                expected_spell_name="Alpha",
                variant_id="quiet",
                variant_name="quiet",
                prompt="Say it clearly but quieter than normal.",
            )
        ],
    )

    loaded = load_calibration_examples(session_dir)

    assert loaded == [
        CalibrationExample(
            path=path,
            kind="positive",
            expected_spell_id="spell-1",
            expected_spell_name="Alpha",
            variant_id="quiet",
            variant_name="quiet",
            prompt="Say it clearly but quieter than normal.",
        )
    ]


def test_diagnose_sweep_counts_variants(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path / "data")
    book, spell = create_spell(book, "Alpha")
    sample_dir = book.data_dir / "samples" / f"spell_{spell.id}"
    sample_dir.mkdir(parents=True)
    for i in range(2):
        path = sample_dir / f"voice_{i + 1:03d}.wav"
        path.touch()
        book = add_voice_sample(book, spell, path.relative_to(book.data_dir).as_posix())

    session_dir = tmp_path / "calibration" / "session"
    examples = []
    for variant_name in ("clean", "fast"):
        attempt = session_dir / "positives" / "alpha" / variant_name / "attempt_001.wav"
        attempt.parent.mkdir(parents=True)
        attempt.touch()
        examples.append(
            CalibrationExample(
                path=attempt,
                kind="positive",
                expected_spell_id=spell.id,
                expected_spell_name=spell.name,
                variant_id=variant_name,
                variant_name=variant_name,
            )
        )
    write_calibration_metadata(session_dir, examples)

    backend = VoiceTemplateBackend(
        name="fake",
        extract_path=lambda _path, _config: np.array([[1.0]], dtype=np.float32),
        extract_array=lambda audio, _config, _sample_rate: audio,
        distance=lambda _a, _b: 0.0,
        aggregate=lambda distances: float(np.median(distances)),
    )

    report = diagnose_calibration_session(
        session_dir, book, VoiceRecognitionConfig(relative_margin_min=0.0), backend
    )

    assert report.sweep[0].variants
    assert [(v.variant_name, v.positive_correct) for v in report.sweep[0].variants] == [
        ("clean", 1),
        ("fast", 1),
    ]


def test_faster_whisper_backend_missing_dependencies_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osc_grimoire.faster_whisper_backends import (
        MissingFasterWhisperDependenciesError,
        _load_faster_whisper_model,
        missing_faster_whisper_dependencies_message,
    )

    _load_faster_whisper_model.cache_clear()
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no faster-whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(MissingFasterWhisperDependenciesError) as exc_info:
        _load_faster_whisper_model("tiny")

    assert str(exc_info.value) == missing_faster_whisper_dependencies_message()
    assert "uv sync" in str(exc_info.value)
    _load_faster_whisper_model.cache_clear()


def test_faster_whisper_feature_extraction_pads_waveform_before_mel() -> None:
    from osc_grimoire.faster_whisper_backends import _extract_whisper_features

    class FakeFeatureExtractor:
        n_samples = 6
        nb_max_frames = 3

        def __call__(self, audio, padding=0):
            assert padding == 0
            assert audio.tolist() == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
            return np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

    features = _extract_whisper_features(
        np.asarray([1.0, 2.0], dtype=np.float32), FakeFeatureExtractor()
    )

    np.testing.assert_array_equal(
        features, np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)
    )


def test_faster_whisper_model_path_prefers_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osc_grimoire.faster_whisper_backends import _resolve_model_path

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("OSC_GRIMOIRE_MODEL_DIR", str(model_dir))

    resolved = _resolve_model_path("tiny")

    assert resolved.model_path == model_dir
    assert resolved.local_files_only


def test_faster_whisper_model_path_uses_repo_vendor_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osc_grimoire.faster_whisper_backends import _resolve_model_path

    model_dir = tmp_path / "vendor" / "models" / "faster-whisper-tiny"
    model_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSC_GRIMOIRE_MODEL_DIR", raising=False)

    resolved = _resolve_model_path("tiny")

    assert resolved.model_path == model_dir
    assert resolved.local_files_only


def test_parakeet_model_path_prefers_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osc_grimoire.parakeet_ctc_backends import _resolve_parakeet_model

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("OSC_GRIMOIRE_PARAKEET_CTC_MODEL_DIR", str(model_dir))

    resolved = _resolve_parakeet_model("example/repo")

    assert resolved.model_dir == model_dir


def test_parakeet_model_path_uses_repo_vendor_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osc_grimoire.parakeet_ctc_backends import _resolve_parakeet_model

    model_dir = tmp_path / "vendor" / "models" / "parakeet-ctc-110m-int8"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").touch()
    (model_dir / "vocab.txt").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSC_GRIMOIRE_PARAKEET_CTC_MODEL_DIR", raising=False)

    resolved = _resolve_parakeet_model("example/repo")

    assert resolved.model_dir == model_dir


def test_nbest_text_normalization_and_similarity() -> None:
    from osc_grimoire.faster_whisper_backends import (
        NBestHypothesis,
        _hypothesis_similarity,
        _normalize_hypothesis_text,
    )

    assert _normalize_hypothesis_text(" Aloha, Mora! ") == "alohamora"
    assert (
        _hypothesis_similarity(
            NBestHypothesis(
                text="Aloha Mora",
                normalized_text="alohamora",
                tokens=(1, 2, 3),
                score=-0.1,
                weight=0.8,
            ),
            NBestHypothesis(
                text="alohomora",
                normalized_text="alohomora",
                tokens=(1, 4, 3),
                score=-0.2,
                weight=0.7,
            ),
        )
        > 0.75
    )


def test_nbest_weighted_distance_prefers_matching_hypotheses() -> None:
    from osc_grimoire.faster_whisper_backends import (
        NBestFeature,
        NBestHypothesis,
        _nbest_distance,
    )

    query = NBestFeature(
        (
            NBestHypothesis("Lumos", "lumos", (10, 20), -0.1, 0.9),
            NBestHypothesis("Lou Moss", "loumoss", (11, 21), -2.0, 0.1),
        )
    )
    matching = NBestFeature(
        (
            NBestHypothesis("Lumos", "lumos", (10, 20), -0.2, 0.8),
            NBestHypothesis("Lou Must", "loumust", (11, 22), -2.1, 0.2),
        )
    )
    different = NBestFeature(
        (NBestHypothesis("Flipendo", "flipendo", (30, 40), -0.1, 1.0),)
    )

    assert _nbest_distance(query, matching) < _nbest_distance(query, different)


def test_nbest_softmax_weights_sum_to_one() -> None:
    from osc_grimoire.faster_whisper_backends import _softmax

    weights = _softmax((-1.0, -2.0, -3.0))

    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_ctc_greedy_token_ids_collapses_repeats_and_blanks() -> None:
    from osc_grimoire.parakeet_ctc_backends import ctc_greedy_token_ids

    log_probs = np.full((7, 4), -20.0, dtype=np.float32)
    for frame, token_id in enumerate((3, 1, 1, 3, 2, 2, 3)):
        log_probs[frame, token_id] = 0.0

    assert ctc_greedy_token_ids(log_probs, blank_id=3) == (1, 2)


def test_ctc_spell_name_text_normalization() -> None:
    from osc_grimoire.parakeet_ctc_backends import (
        ctc_token_ids_to_text,
        normalize_spoken_spell_name,
    )

    text = ctc_token_ids_to_text(
        (1, 2, 3, 4, 5),
        {
            1: " alo",
            2: "ho",
            3: "<blk>",
            4: "mora!",
            5: "  ",
        },
    )

    assert normalize_spoken_spell_name(text) == "Alohomora"
    assert normalize_spoken_spell_name("...") == ""


def test_ctc_sequence_log_probability_prefers_matching_sequence() -> None:
    from osc_grimoire.parakeet_ctc_backends import ctc_sequence_log_probability

    probs = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    log_probs = np.log(probs)

    matching = ctc_sequence_log_probability(log_probs, (0, 1), blank_id=2)
    mismatched = ctc_sequence_log_probability(log_probs, (1, 0), blank_id=2)

    assert matching > mismatched


def test_ctc_forced_distance_prefers_template_sequence() -> None:
    from osc_grimoire.parakeet_ctc_backends import CtcFeature, _ctc_forced_distance

    probs = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    query = CtcFeature(np.log(probs), token_ids=(0, 1))
    matching = CtcFeature(np.log(probs), token_ids=(0, 1))
    mismatched = CtcFeature(np.log(probs), token_ids=(1, 0))

    assert _ctc_forced_distance(query, matching) < _ctc_forced_distance(
        query, mismatched
    )
