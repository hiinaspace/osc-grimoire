"""Optional regression test against committed real-voice fixtures.

These clips are the user's own training samples. The active recognizer uses a
bundled/downloaded model, so these tests are opt-in to keep normal unit tests
offline and fast.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.faster_whisper_backends import faster_whisper_dtw_backend
from osc_grimoire.spellbook import (
    Spellbook,
    add_voice_sample,
    create_spell,
    load_spellbook,
)
from osc_grimoire.voice_recognizer import (
    decide,
    leave_one_out_eval,
    rank_spells,
    recompute_all,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "voice"
NEGATIVES_ROOT = Path(__file__).parent / "fixtures" / "voice_negatives"
AUDIO_EXTENSIONS = (".flac", ".wav")


def _audio_files(root: Path, pattern_prefix: str = "") -> list[Path]:
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.suffix.casefold() in AUDIO_EXTENSIONS
        and p.name.startswith(pattern_prefix)
    )


def _build_spellbook_from_fixtures(tmp_path: Path) -> Spellbook:
    """Copy fixture clips into tmp_path and build a real Spellbook from them."""
    book = load_spellbook(tmp_path)
    for spell_dir in sorted(p for p in FIXTURE_ROOT.iterdir() if p.is_dir()):
        clips = _audio_files(spell_dir, "voice_")
        if not clips:
            continue
        book, spell = create_spell(book, spell_dir.name)
        target_dir = tmp_path / "samples" / f"spell_{spell.id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in clips:
            dst = target_dir / src.name
            shutil.copyfile(src, dst)
            rel = dst.relative_to(tmp_path).as_posix()
            current = next(s for s in book.spells if s.id == spell.id)
            book = add_voice_sample(book, current, rel)
    return book


def _has_fixtures() -> bool:
    return FIXTURE_ROOT.exists() and any(FIXTURE_ROOT.iterdir())


def _run_model_fixture_tests() -> bool:
    return os.environ.get("OSC_GRIMOIRE_RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(
    not (_has_fixtures() and _run_model_fixture_tests()),
    reason="voice model fixture tests are opt-in",
)
def test_leave_one_out_classifies_all_fixtures(tmp_path: Path) -> None:
    book = _build_spellbook_from_fixtures(tmp_path)
    config = VoiceRecognitionConfig()
    backend = faster_whisper_dtw_backend()
    book = recompute_all(book, config, backend)

    results = leave_one_out_eval(book, config, backend)
    assert results, "fixtures present but no LOO results produced"

    incorrect = [r for r in results if not r.correct]
    assert not incorrect, (
        "Expected every held-out fixture to classify correctly. "
        f"Misses: {[(r.spell_name, r.best_spell_name, r.sample_path.name) for r in incorrect]}"
    )

    # All known-positive samples should also pass both gates with default
    # thresholds; this is the regression we actually care about.
    for r in results:
        assert r.intra_ratio is not None
        assert r.intra_ratio < config.intra_class_ratio_max, (
            f"{r.spell_name}/{r.sample_path.name} intra_ratio={r.intra_ratio:.2f} "
            f">= {config.intra_class_ratio_max}"
        )
        if r.margin_ratio is not None:
            assert r.margin_ratio >= config.relative_margin_min, (
                f"{r.spell_name}/{r.sample_path.name} margin_ratio={r.margin_ratio:.2f} "
                f"< {config.relative_margin_min}"
            )


@pytest.mark.skipif(
    not (_has_fixtures() and _run_model_fixture_tests()),
    reason="voice model fixture tests are opt-in",
)
def test_query_each_fixture_recognizes_self(tmp_path: Path) -> None:
    """A non-LOO sanity check: querying a fixture against the full spellbook
    (which contains that fixture) must rank its own spell first with both gates
    passing easily."""
    book = _build_spellbook_from_fixtures(tmp_path)
    config = VoiceRecognitionConfig()
    backend = faster_whisper_dtw_backend()
    book = recompute_all(book, config, backend)

    for spell_dir in sorted(p for p in FIXTURE_ROOT.iterdir() if p.is_dir()):
        clips = _audio_files(spell_dir, "voice_")
        for clip in clips:
            query = backend.extract_path(clip, config)
            ranking = rank_spells(query, book, config, backend=backend)
            assert ranking, "no rankable spells"
            assert ranking[0].name == spell_dir.name, (
                f"{clip.name}: ranked {ranking[0].name} first, expected {spell_dir.name}"
            )
            decision = decide(ranking, config)
            assert decision.accepted, (
                f"{spell_dir.name}/{clip.name} unexpectedly rejected: {decision.reason}"
            )


def _has_negatives() -> bool:
    return NEGATIVES_ROOT.exists() and bool(_audio_files(NEGATIVES_ROOT))


@pytest.mark.skipif(
    not (_has_fixtures() and _has_negatives() and _run_model_fixture_tests()),
    reason="voice model fixture tests are opt-in",
)
def test_negatives_are_rejected(tmp_path: Path) -> None:
    book = _build_spellbook_from_fixtures(tmp_path)
    config = VoiceRecognitionConfig()
    backend = faster_whisper_dtw_backend()
    book = recompute_all(book, config, backend)

    accepted_negatives = []
    for clip in _audio_files(NEGATIVES_ROOT):
        query = backend.extract_path(clip, config)
        ranking = rank_spells(query, book, config, backend=backend)
        decision = decide(ranking, config)
        if decision.accepted:
            accepted_negatives.append(
                (
                    clip.name,
                    ranking[0].name,
                    decision.intra_ratio,
                    decision.margin_ratio,
                )
            )

    assert not accepted_negatives, (
        f"Expected all negatives to be rejected but {len(accepted_negatives)} "
        f"were accepted: {accepted_negatives}"
    )
