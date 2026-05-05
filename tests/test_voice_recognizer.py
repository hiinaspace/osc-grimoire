from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import Spellbook, create_spell
from osc_grimoire.voice_recognizer import (
    AliasRanking,
    SpellRanking,
    TextHypothesis,
    VoiceTemplateBackend,
    decide,
    rank_spells,
    text_hypotheses,
)


def _ranking(name: str, agg: float) -> SpellRanking:
    tokens = _tokenize(name)
    return SpellRanking(
        spell_id=name.lower(),
        name=name,
        aggregate_distance=agg,
        alias=name,
        token_ids=tokens,
        alias_rankings=(AliasRanking(name, agg, tokens),),
    )


def _tokenize(text: str) -> tuple[int, ...]:
    cleaned = "".join(c for c in text.casefold() if c.isalnum())
    if not cleaned:
        raise ValueError("empty")
    return tuple(ord(c) for c in cleaned)


def _fake_backend() -> VoiceTemplateBackend:
    def distance_to_tokens(feature, token_ids: tuple[int, ...]) -> float:
        return (
            0.0
            if tuple(feature) == token_ids
            else float(abs(sum(feature) - sum(token_ids)) + 1)
        )

    return VoiceTemplateBackend(
        name="fake",
        extract_path=lambda path, _config: _tokenize(path.stem),
        extract_array=lambda audio, _config, _sample_rate: tuple(
            int(v) for v in np.asarray(audio).reshape(-1)
        ),
        distance=lambda _a, _b: 0.0,
        aggregate=lambda distances: float(np.median(distances)),
        tokenize_text=_tokenize,
        distance_to_tokens=distance_to_tokens,
        text_hypotheses=lambda _feature, _limit: (
            TextHypothesis("Alpha", _tokenize("Alpha"), -1.0),
        ),
    )


def test_rank_spells_identifies_matching_alias(tmp_path: Path) -> None:
    book = Spellbook(tmp_path)
    book, spell_a = create_spell(book, "Alpha")
    book, spell_b = create_spell(book, "Bravo")
    backend = _fake_backend()

    rankings = rank_spells(
        _tokenize("Alpha"), book, VoiceRecognitionConfig(), backend=backend
    )

    assert rankings[0].spell_id == spell_a.id
    assert rankings[0].alias == "Alpha"
    assert rankings[0].aggregate_distance == 0.0
    assert rankings[0].alias_rankings[0].alias == "Alpha"
    assert rankings[0].alias_rankings[0].distance == 0.0
    assert rankings[1].spell_id == spell_b.id
    assert rankings[1].aggregate_distance > 0.0


def test_rank_spells_keeps_all_scored_aliases(tmp_path: Path) -> None:
    from osc_grimoire.spellbook import add_voice_alias

    book = Spellbook(tmp_path)
    book, spell = create_spell(book, "Alpha")
    book = add_voice_alias(book, spell, "Al fa")

    rankings = rank_spells(
        _tokenize("Al fa"), book, VoiceRecognitionConfig(), backend=_fake_backend()
    )

    assert [row.alias for row in rankings[0].alias_rankings] == ["Al fa", "Alpha"]


def test_rank_spells_skips_spell_with_no_incantations(tmp_path: Path) -> None:
    from osc_grimoire.spellbook import remove_voice_alias

    book = Spellbook(tmp_path)
    book, spell = create_spell(book, "Alpha")
    book = remove_voice_alias(book, spell, "Alpha")

    rankings = rank_spells(
        _tokenize("Alpha"), book, VoiceRecognitionConfig(), backend=_fake_backend()
    )

    assert rankings == []


def test_rank_spells_requires_text_backend(tmp_path: Path) -> None:
    book = Spellbook(tmp_path)
    book, _spell = create_spell(book, "Alpha")
    backend = VoiceTemplateBackend(
        name="no-text",
        extract_path=lambda _path, _config: (),
        extract_array=lambda _audio, _config, _sample_rate: (),
        distance=lambda _a, _b: 0.0,
        aggregate=lambda distances: float(np.median(distances)),
    )

    with pytest.raises(RuntimeError, match="cannot score text incantations"):
        rank_spells((), book, VoiceRecognitionConfig(), backend=backend)


def test_decide_rejects_when_alias_distance_too_high() -> None:
    rankings = [_ranking("Alpha", 8.0), _ranking("Bravo", 20.0)]
    config = VoiceRecognitionConfig(
        voice_alias_distance_max=7.0, relative_margin_min=0.0
    )
    decision = decide(rankings, config)
    assert decision.accepted is False
    assert "incantation distance" in decision.reason


def test_decide_accepts_when_both_gates_pass() -> None:
    rankings = [_ranking("Alpha", 3.0), _ranking("Bravo", 10.0)]
    config = VoiceRecognitionConfig(
        voice_alias_distance_max=7.0, relative_margin_min=0.2
    )
    decision = decide(rankings, config)
    assert decision.accepted is True
    assert decision.best_distance == pytest.approx(3.0)
    assert decision.margin_ratio is not None and decision.margin_ratio > 0.2


def test_decide_rejects_when_margin_too_thin() -> None:
    rankings = [_ranking("Alpha", 3.0), _ranking("Bravo", 3.1)]
    config = VoiceRecognitionConfig(
        voice_alias_distance_max=7.0, relative_margin_min=0.20
    )
    decision = decide(rankings, config)
    assert decision.accepted is False
    assert "relative margin" in decision.reason


def test_decide_skips_margin_gate_with_one_spell() -> None:
    rankings = [_ranking("Solo", 6.0)]
    config = VoiceRecognitionConfig(
        voice_alias_distance_max=7.0, relative_margin_min=0.99
    )
    decision = decide(rankings, config)
    assert decision.margin_ratio is None
    assert decision.accepted is True


def test_text_hypotheses_uses_backend_support() -> None:
    hypotheses = text_hypotheses((), _fake_backend())

    assert hypotheses[0].text == "Alpha"
