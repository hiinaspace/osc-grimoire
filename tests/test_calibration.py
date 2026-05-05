from __future__ import annotations

from pathlib import Path

import numpy as np

from osc_grimoire.calibration import (
    CalibrationExample,
    diagnose_calibration_session,
    load_calibration_examples,
    write_calibration_metadata,
)
from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import Spellbook, create_spell
from osc_grimoire.voice_recognizer import TextHypothesis, VoiceTemplateBackend


def _tokenize(text: str) -> tuple[int, ...]:
    cleaned = "".join(c for c in text.casefold() if c.isalnum())
    if not cleaned:
        raise ValueError("empty")
    return tuple(ord(c) for c in cleaned)


def _text(tokens: tuple[int, ...]) -> str:
    return "".join(chr(t) for t in tokens)


def _fake_backend() -> VoiceTemplateBackend:
    def distance_to_tokens(feature: str, token_ids: tuple[int, ...]) -> float:
        alias = _text(token_ids)
        if feature.startswith(alias):
            return 1.0
        if feature.startswith("negative"):
            return 20.0
        return 10.0

    return VoiceTemplateBackend(
        name="fake",
        extract_path=lambda path, _config: path.stem.casefold(),
        extract_array=lambda _audio, _config, _sample_rate: "alpha",
        distance=lambda _a, _b: 0.0,
        aggregate=lambda distances: float(np.median(distances)),
        tokenize_text=_tokenize,
        distance_to_tokens=distance_to_tokens,
        text_hypotheses=lambda _feature, _limit: (
            TextHypothesis("Alpha", _tokenize("Alpha"), -1.0),
        ),
    )


def _book(tmp_path: Path) -> tuple[Spellbook, str]:
    book = Spellbook(tmp_path)
    book, spell = create_spell(book, "Alpha")
    return book, spell.id


def test_diagnose_calibration_session_recommends_margin(tmp_path: Path) -> None:
    book, spell_id = _book(tmp_path)
    session_dir = tmp_path / "calibration" / "session"
    positive = session_dir / "positives" / "alpha" / "alpha_001.wav"
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
                expected_spell_id=spell_id,
                expected_spell_name="Alpha",
            ),
            CalibrationExample(path=negative, kind="negative"),
        ],
    )

    report = diagnose_calibration_session(
        session_dir,
        book,
        VoiceRecognitionConfig(voice_alias_distance_max=7.0),
        _fake_backend(),
    )

    assert report.recommended_margin_min is not None
    assert report.sweep[0].positive_correct == 1
    assert report.sweep[0].negative_accepted == 0
    assert report.examples[0].best_alias == "Alpha"
    assert report.examples[0].text_hypotheses[0].text == "Alpha"


def test_diagnose_ignores_spells_not_labeled_in_session(tmp_path: Path) -> None:
    book = Spellbook(tmp_path)
    book, expected = create_spell(book, "Expected")
    book, _extra = create_spell(book, "Extra")
    session_dir = tmp_path / "calibration" / "session"
    positive = session_dir / "positives" / "expected" / "expected_001.wav"
    positive.parent.mkdir(parents=True)
    positive.touch()
    write_calibration_metadata(
        session_dir,
        [
            CalibrationExample(
                path=positive,
                kind="positive",
                expected_spell_id=expected.id,
                expected_spell_name=expected.name,
            )
        ],
    )

    report = diagnose_calibration_session(
        session_dir,
        book,
        VoiceRecognitionConfig(),
        _fake_backend(),
    )

    assert report.examples[0].best_spell_id == expected.id


def test_calibration_metadata_round_trips_variant_fields(tmp_path: Path) -> None:
    session_dir = tmp_path / "calibration" / "session"
    path = session_dir / "positives" / "alpha" / "quiet" / "attempt_001.wav"
    path.parent.mkdir(parents=True)
    path.touch()
    examples = [
        CalibrationExample(
            path=path,
            kind="positive",
            expected_spell_id="spell-1",
            expected_spell_name="Alpha",
            variant_id="quiet",
            variant_name="quiet",
            prompt="Say it quietly.",
        )
    ]

    write_calibration_metadata(session_dir, examples)

    assert load_calibration_examples(session_dir) == examples


def test_diagnose_sweep_counts_variants(tmp_path: Path) -> None:
    book, spell_id = _book(tmp_path)
    session_dir = tmp_path / "calibration" / "session"
    examples: list[CalibrationExample] = []
    for variant_name in ("clean", "quiet"):
        attempt = session_dir / "positives" / "alpha" / variant_name / "alpha.wav"
        attempt.parent.mkdir(parents=True)
        attempt.touch()
        examples.append(
            CalibrationExample(
                path=attempt,
                kind="positive",
                expected_spell_id=spell_id,
                expected_spell_name="Alpha",
                variant_name=variant_name,
            )
        )
    write_calibration_metadata(session_dir, examples)

    report = diagnose_calibration_session(
        session_dir, book, VoiceRecognitionConfig(), _fake_backend()
    )

    assert [(v.variant_name, v.positive_correct) for v in report.sweep[0].variants] == [
        ("clean", 1),
        ("quiet", 1),
    ]


def test_ctc_spell_name_text_normalization() -> None:
    from osc_grimoire.parakeet_ctc_backends import (
        ctc_token_ids_to_text,
        normalize_spoken_spell_name,
    )

    text = ctc_token_ids_to_text(
        (1, 2, 3),
        {
            1: " Alo",
            2: "ho",
            3: "mora.",
        },
    )

    assert normalize_spoken_spell_name(text) == "Alohomora"


def test_ctc_sequence_log_probability_prefers_matching_sequence() -> None:
    from osc_grimoire.parakeet_ctc_backends import ctc_sequence_log_probability

    probs = np.asarray(
        [
            [0.80, 0.05, 0.15],
            [0.05, 0.80, 0.15],
            [0.05, 0.80, 0.15],
            [0.05, 0.05, 0.90],
        ],
        dtype=np.float32,
    )
    matching = ctc_sequence_log_probability(np.log(probs), (0, 1), blank_id=2)
    mismatched = ctc_sequence_log_probability(np.log(probs), (1, 0), blank_id=2)

    assert matching > mismatched


def test_ctc_prefix_beam_returns_matching_text() -> None:
    from osc_grimoire.parakeet_ctc_backends import ctc_prefix_beam_search

    probs = np.asarray(
        [
            [0.80, 0.05, 0.15],
            [0.05, 0.80, 0.15],
            [0.05, 0.05, 0.90],
        ],
        dtype=np.float32,
    )

    beams = ctc_prefix_beam_search(np.log(probs), blank_id=2, beam_size=3)

    assert beams[0][0] == (0, 1)
