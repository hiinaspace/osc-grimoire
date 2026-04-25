from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osc_grimoire.calibration import (
    CalibrationExample,
    diagnose_calibration_session,
    write_calibration_metadata,
)
from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.voice_recognizer import recompute_all
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
