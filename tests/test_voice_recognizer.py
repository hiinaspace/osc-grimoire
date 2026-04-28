from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osc_grimoire.config import VoiceRecognitionConfig
from osc_grimoire.spellbook import (
    add_voice_sample,
    create_spell,
    load_spellbook,
    voice_sample_abs_paths,
)
from osc_grimoire.voice_recognizer import (
    SpellRanking,
    VoiceTemplateBackend,
    compute_intra_class_median,
    decide,
    rank_spells,
    recompute_spell_voice_stats,
)


def _ranking(name: str, agg: float, intra: float | None) -> SpellRanking:
    return SpellRanking(
        spell_id=name.lower(),
        name=name,
        aggregate_distance=agg,
        per_sample_distances=(agg,),
        intra_class_median=intra,
    )


def _stub_features(seed: int, frames: int = 64, dimensions: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((frames, dimensions)).astype(np.float32)


def _fake_backend() -> VoiceTemplateBackend:
    return VoiceTemplateBackend(
        name="fake",
        extract_path=lambda _path, _config: _stub_features(0),
        extract_array=lambda audio, _config, _sample_rate: audio,
        distance=lambda a, b: float(np.linalg.norm(a - b)),
        aggregate=lambda distances: float(np.median(distances)),
    )


def _spell_with_samples(book, name: str, n_samples: int, sample_features: np.ndarray):
    book, spell = create_spell(book, name)
    cache: dict[Path, np.ndarray] = {}
    for i in range(n_samples):
        rel = f"samples/spell_{spell.id}/voice_{i + 1:03d}.wav"
        abs_path = book.data_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.touch()
        current = next(s for s in book.spells if s.id == spell.id)
        book = add_voice_sample(book, current, rel)
        cache[abs_path] = sample_features
    return book, spell, cache


def test_rank_spells_identifies_correct_spell(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    feat_a = _stub_features(seed=1)
    feat_b = _stub_features(seed=2)

    book, spell_a, cache_a = _spell_with_samples(book, "Alpha", 3, feat_a)
    book, spell_b, cache_b = _spell_with_samples(book, "Bravo", 3, feat_b)
    cache = {**cache_a, **cache_b}

    config = VoiceRecognitionConfig()

    rankings = rank_spells(
        feat_a, book, config, feature_cache=cache, backend=_fake_backend()
    )
    assert rankings[0].spell_id == spell_a.id
    assert rankings[0].aggregate_distance == 0.0
    assert rankings[1].spell_id == spell_b.id
    assert rankings[1].aggregate_distance > 0.0


def test_compute_intra_class_median_returns_none_for_one_sample() -> None:
    assert compute_intra_class_median([_stub_features(0)], _fake_backend()) is None


def test_compute_intra_class_median_positive_for_distinct_samples() -> None:
    feats = [_stub_features(i) for i in range(4)]
    median = compute_intra_class_median(feats, _fake_backend())
    assert median is not None
    assert median > 0.0


def test_decide_rejects_when_intra_ratio_too_high() -> None:
    rankings = [
        _ranking("Alpha", agg=100.0, intra=10.0),
        _ranking("Bravo", agg=500.0, intra=10.0),
    ]
    config = VoiceRecognitionConfig(intra_class_ratio_max=2.0, relative_margin_min=0.0)
    decision = decide(rankings, config)
    assert decision.accepted is False
    assert "intra-class ratio" in decision.reason


def test_decide_accepts_when_both_gates_pass() -> None:
    rankings = [
        _ranking("Alpha", agg=12.0, intra=10.0),
        _ranking("Bravo", agg=80.0, intra=10.0),
    ]
    config = VoiceRecognitionConfig(intra_class_ratio_max=2.0, relative_margin_min=0.2)
    decision = decide(rankings, config)
    assert decision.accepted is True
    assert decision.intra_ratio is not None and decision.intra_ratio < 2.0
    assert decision.margin_ratio is not None and decision.margin_ratio > 0.2


def test_decide_rejects_when_margin_too_thin() -> None:
    rankings = [
        _ranking("Alpha", agg=100.0, intra=100.0),
        _ranking("Bravo", agg=105.0, intra=100.0),  # very close
    ]
    config = VoiceRecognitionConfig(intra_class_ratio_max=5.0, relative_margin_min=0.20)
    decision = decide(rankings, config)
    assert decision.accepted is False
    assert "relative margin" in decision.reason


def test_decide_skips_margin_gate_with_one_spell() -> None:
    rankings = [_ranking("Solo", agg=10.0, intra=20.0)]
    config = VoiceRecognitionConfig(intra_class_ratio_max=2.0, relative_margin_min=0.99)
    decision = decide(rankings, config)
    assert decision.margin_ratio is None
    assert decision.accepted is True


def test_decide_uses_fallback_when_intra_unknown() -> None:
    rankings = [_ranking("NoIntra", agg=300.0, intra=None)]
    config = VoiceRecognitionConfig(
        intra_class_ratio_max=2.0,
        relative_margin_min=0.0,
        untrained_distance_fallback=100.0,
    )
    decision = decide(rankings, config)
    assert decision.accepted is False  # 300 / 100 = 3.0 > 2.0
    assert decision.intra_ratio == pytest.approx(3.0)


def test_voice_sample_abs_paths_resolves_relative(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    book, spell = create_spell(book, "Probe")
    book = add_voice_sample(book, spell, "samples/x/voice_001.wav")
    fresh = next(s for s in book.spells if s.id == spell.id)
    paths = voice_sample_abs_paths(book, fresh)
    assert paths == [tmp_path / "samples" / "x" / "voice_001.wav"]


def test_recompute_spell_voice_stats_with_synthetic_features(tmp_path: Path) -> None:
    book = load_spellbook(tmp_path)
    feat = _stub_features(seed=1)
    feat2 = _stub_features(seed=2)
    book, spell = create_spell(book, "Alpha")
    cache: dict[Path, np.ndarray] = {}
    for i, f in enumerate([feat, feat2, feat]):
        rel = f"samples/spell_{spell.id}/voice_{i + 1:03d}.wav"
        abs_path = book.data_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.touch()
        current = next(s for s in book.spells if s.id == spell.id)
        book = add_voice_sample(book, current, rel)
        cache[abs_path] = f

    book = recompute_spell_voice_stats(
        book,
        spell,
        VoiceRecognitionConfig(),
        feature_cache=cache,
        backend=_fake_backend(),
    )
    fresh = next(s for s in book.spells if s.id == spell.id)
    assert fresh.intra_class_median is not None
    assert fresh.intra_class_median > 0.0
