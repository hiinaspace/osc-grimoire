from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from osc_grimoire.calibration import (
    CalibrationExample,
    diagnose_calibration_session,
    write_calibration_metadata,
)
from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import add_voice_sample, create_spell, load_spellbook
from osc_grimoire.voice_recognizer import VoiceTemplateBackend, recompute_all
from tests.test_voice_fixtures import (
    FIXTURE_ROOT,
    NEGATIVES_ROOT,
    _audio_files,
    _build_spellbook_from_fixtures,
    _has_fixtures,
    _has_negatives,
)


@pytest.mark.skipif(
    not (_has_fixtures() and _has_negatives()),
    reason="no voice fixtures committed",
)
def test_diagnose_calibration_session_recommends_margin(tmp_path: Path) -> None:
    book = _build_spellbook_from_fixtures(tmp_path / "data")
    config = VoiceRecognitionConfig()
    book = recompute_all(book, config)

    session_dir = tmp_path / "calibration" / "session"
    examples: list[CalibrationExample] = []

    for spell in book.spells:
        spell_fixture_dir = FIXTURE_ROOT / spell.name
        target_dir = session_dir / "positives" / spell.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in _audio_files(spell_fixture_dir, "voice_")[:2]:
            dst = target_dir / src.name
            shutil.copyfile(src, dst)
            examples.append(
                CalibrationExample(
                    path=dst,
                    kind="positive",
                    expected_spell_id=spell.id,
                    expected_spell_name=spell.name,
                )
            )

    negative_dir = session_dir / "negatives"
    negative_dir.mkdir(parents=True, exist_ok=True)
    for src in _audio_files(NEGATIVES_ROOT)[:4]:
        dst = negative_dir / src.name
        shutil.copyfile(src, dst)
        examples.append(CalibrationExample(path=dst, kind="negative"))

    write_calibration_metadata(session_dir, examples)
    report = diagnose_calibration_session(session_dir, book, config)

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


def test_embedding_backend_missing_dependencies_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osc_grimoire.voice_embedding_backends import (
        MissingEmbeddingDependenciesError,
        _load_model,
        missing_embedding_dependencies_message,
    )

    _load_model.cache_clear()
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(MissingEmbeddingDependenciesError) as exc_info:
        _load_model("example/model")

    assert str(exc_info.value) == missing_embedding_dependencies_message()
    assert "uv sync --group ml" in str(exc_info.value)
    _load_model.cache_clear()
