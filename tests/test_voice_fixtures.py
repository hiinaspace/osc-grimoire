from __future__ import annotations

from pathlib import Path

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import Spellbook, create_spell
from osc_grimoire.voice_recognizer import decide, rank_spells, text_hypotheses

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
VOICE_ROOT = FIXTURE_ROOT / "voice"
NEGATIVE_ROOT = FIXTURE_ROOT / "voice_negatives"


def _fixture_spellbook(tmp_path: Path) -> Spellbook:
    book = Spellbook(tmp_path)
    for name in ("alohomora", "flipendo", "lumos"):
        book, _spell = create_spell(book, name)
    return book


def test_alias_recognizer_classifies_fixture_audio(tmp_path: Path) -> None:
    from osc_grimoire.parakeet_ctc_backends import parakeet_ctc_forced_backend

    backend = parakeet_ctc_forced_backend()
    config = VoiceRecognitionConfig(voice_alias_distance_max=7.0)
    book = _fixture_spellbook(tmp_path)

    for spell_dir in sorted(VOICE_ROOT.iterdir()):
        if not spell_dir.is_dir():
            continue
        for path in sorted(spell_dir.glob("*.flac")):
            query = backend.extract_path(path, config)
            ranking = rank_spells(query, book, config, backend=backend)
            decision = decide(ranking, config)
            assert ranking[0].name.casefold() == spell_dir.name
            assert decision.accepted, (
                f"{spell_dir.name}/{path.name} rejected: {decision.reason}"
            )


def test_alias_recognizer_rejects_negative_fixtures(tmp_path: Path) -> None:
    from osc_grimoire.parakeet_ctc_backends import parakeet_ctc_forced_backend

    backend = parakeet_ctc_forced_backend()
    config = VoiceRecognitionConfig(voice_alias_distance_max=7.0)
    book = _fixture_spellbook(tmp_path)

    false_accepts: list[tuple[str, str, float | None]] = []
    for path in sorted(NEGATIVE_ROOT.glob("*.flac")):
        query = backend.extract_path(path, config)
        ranking = rank_spells(query, book, config, backend=backend)
        decision = decide(ranking, config)
        if decision.accepted:
            false_accepts.append((path.name, ranking[0].name, decision.best_distance))

    assert false_accepts == []


def test_parakeet_text_hypotheses_are_available_for_fixture_audio() -> None:
    from osc_grimoire.parakeet_ctc_backends import parakeet_ctc_forced_backend

    backend = parakeet_ctc_forced_backend()
    config = VoiceRecognitionConfig()
    query = backend.extract_path(VOICE_ROOT / "alohomora" / "voice_001.flac", config)

    hypotheses = text_hypotheses(query, backend)

    assert hypotheses
    assert any("Alo" in h.text for h in hypotheses)
