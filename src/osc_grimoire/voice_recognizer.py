from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

from .config import VoiceRecognitionConfig
from .spellbook import (
    Spell,
    Spellbook,
    replace_spell,
    voice_sample_abs_paths,
)
from .voice_features import FloatArray

VoiceFeature = Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceTemplateBackend:
    name: str
    extract_path: Callable[[Path, VoiceRecognitionConfig], VoiceFeature]
    extract_array: Callable[[FloatArray, VoiceRecognitionConfig, int], VoiceFeature]
    distance: Callable[[VoiceFeature, VoiceFeature], float]
    aggregate: Callable[[list[float]], float]


@dataclass(frozen=True)
class SpellRanking:
    spell_id: str
    name: str
    aggregate_distance: float
    per_sample_distances: tuple[float, ...]
    intra_class_median: float | None


@dataclass(frozen=True)
class BackendStats:
    intra_class_medians: dict[str, float | None]
    extraction_seconds: float = 0.0


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    intra_ratio: float | None
    intra_ratio_max: float
    margin_ratio: float | None
    margin_ratio_min: float


@dataclass(frozen=True)
class LooResult:
    spell_id: str
    spell_name: str
    sample_path: Path
    best_spell_id: str
    best_spell_name: str
    best_distance: float
    second_distance: float | None
    intra_ratio: float | None
    margin_ratio: float | None
    correct: bool


def rank_spells(
    query: VoiceFeature,
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    feature_cache: dict[Path, VoiceFeature] | None = None,
    backend: VoiceTemplateBackend | None = None,
    backend_stats: BackendStats | None = None,
) -> list[SpellRanking]:
    backend = backend or default_voice_backend()
    rankings: list[SpellRanking] = []
    for spell in spellbook.spells:
        if not spell.enabled or not spell.has_voice:
            continue
        sample_paths = voice_sample_abs_paths(spellbook, spell)
        if not sample_paths:
            continue
        per_sample = _distances_to_samples(
            query, sample_paths, config, feature_cache, backend
        )
        if not per_sample:
            continue
        rankings.append(
            SpellRanking(
                spell_id=spell.id,
                name=spell.name,
                aggregate_distance=backend.aggregate(per_sample),
                per_sample_distances=tuple(per_sample),
                intra_class_median=_intra_class_median_for(spell, backend_stats),
            )
        )
    rankings.sort(key=lambda r: r.aggregate_distance)
    return rankings


def _intra_class_median_for(
    spell: Spell, backend_stats: BackendStats | None
) -> float | None:
    if backend_stats is None:
        return spell.intra_class_median
    return backend_stats.intra_class_medians.get(spell.id)


def decide(
    rankings: list[SpellRanking],
    config: VoiceRecognitionConfig,
) -> Decision:
    if not rankings:
        return Decision(
            accepted=False,
            reason="no enabled spells with voice samples",
            intra_ratio=None,
            intra_ratio_max=config.intra_class_ratio_max,
            margin_ratio=None,
            margin_ratio_min=config.relative_margin_min,
        )

    best = rankings[0]
    second = rankings[1] if len(rankings) > 1 else None

    intra_baseline = (
        best.intra_class_median
        if best.intra_class_median is not None
        else config.untrained_distance_fallback
    )
    intra_ratio = best.aggregate_distance / max(intra_baseline, 1e-9)

    margin_ratio: float | None
    if second is None or second.aggregate_distance <= 0.0:
        margin_ratio = None
    else:
        margin_ratio = (
            second.aggregate_distance - best.aggregate_distance
        ) / second.aggregate_distance

    if intra_ratio > config.intra_class_ratio_max:
        return Decision(
            accepted=False,
            reason=(
                f"intra-class ratio {intra_ratio:.2f} exceeds "
                f"{config.intra_class_ratio_max:.2f} "
                f"(best d={best.aggregate_distance:.1f}, "
                f"intra-median={intra_baseline:.1f})"
            ),
            intra_ratio=intra_ratio,
            intra_ratio_max=config.intra_class_ratio_max,
            margin_ratio=margin_ratio,
            margin_ratio_min=config.relative_margin_min,
        )

    if margin_ratio is not None and margin_ratio < config.relative_margin_min:
        return Decision(
            accepted=False,
            reason=(
                f"relative margin {margin_ratio:.2f} below "
                f"{config.relative_margin_min:.2f} "
                f"(too close to {second.name if second else '?'})"
            ),
            intra_ratio=intra_ratio,
            intra_ratio_max=config.intra_class_ratio_max,
            margin_ratio=margin_ratio,
            margin_ratio_min=config.relative_margin_min,
        )

    return Decision(
        accepted=True,
        reason="both gates pass",
        intra_ratio=intra_ratio,
        intra_ratio_max=config.intra_class_ratio_max,
        margin_ratio=margin_ratio,
        margin_ratio_min=config.relative_margin_min,
    )


def compute_intra_class_median(
    features_list: list[VoiceFeature],
    backend: VoiceTemplateBackend | None = None,
) -> float | None:
    backend = backend or default_voice_backend()
    if len(features_list) < 2:
        return None
    distances: list[float] = []
    for a, b in combinations(features_list, 2):
        distances.append(backend.distance(a, b))
    return float(statistics.median(distances))


def recompute_spell_voice_stats(
    spellbook: Spellbook,
    spell: Spell,
    config: VoiceRecognitionConfig,
    feature_cache: dict[Path, VoiceFeature] | None = None,
    backend: VoiceTemplateBackend | None = None,
) -> Spellbook:
    backend = backend or default_voice_backend()
    # Look up the current spell by id so a stale `spell` value doesn't wipe
    # out voice_samples on substitution.
    current = next((s for s in spellbook.spells if s.id == spell.id), None)
    if current is None:
        return spellbook
    sample_paths = voice_sample_abs_paths(spellbook, current)
    features: list[VoiceFeature] = []
    for path in sample_paths:
        if not path.exists():
            LOGGER.warning("Sample missing on disk: %s", path)
            continue
        if feature_cache is not None and path in feature_cache:
            features.append(feature_cache[path])
            continue
        feats = backend.extract_path(path, config)
        if feature_cache is not None:
            feature_cache[path] = feats
        features.append(feats)

    intra = compute_intra_class_median(features, backend)
    updated = replace(current, intra_class_median=intra)
    return replace_spell(spellbook, updated)


def recompute_all(
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    backend: VoiceTemplateBackend | None = None,
) -> Spellbook:
    backend = backend or default_voice_backend()
    feature_cache: dict[Path, VoiceFeature] = {}
    for spell in list(spellbook.spells):
        if not spell.has_voice or not spell.voice_samples:
            continue
        spellbook = recompute_spell_voice_stats(
            spellbook, spell, config, feature_cache, backend
        )
    return spellbook


def compute_backend_stats(
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    backend: VoiceTemplateBackend | None = None,
) -> tuple[BackendStats, dict[Path, VoiceFeature]]:
    backend = backend or default_voice_backend()
    feature_cache: dict[Path, VoiceFeature] = {}
    intra_class_medians: dict[str, float | None] = {}
    extraction_seconds = 0.0
    for spell in spellbook.spells:
        if not spell.has_voice or not spell.voice_samples:
            continue
        features: list[VoiceFeature] = []
        for path in voice_sample_abs_paths(spellbook, spell):
            if not path.exists():
                LOGGER.warning("Sample missing on disk: %s", path)
                continue
            if path not in feature_cache:
                start = time.perf_counter()
                feature_cache[path] = backend.extract_path(path, config)
                extraction_seconds += time.perf_counter() - start
            features.append(feature_cache[path])
        intra_class_medians[spell.id] = compute_intra_class_median(features, backend)
    return BackendStats(intra_class_medians, extraction_seconds), feature_cache


def leave_one_out_eval(
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    backend: VoiceTemplateBackend | None = None,
) -> list[LooResult]:
    backend = backend or default_voice_backend()
    feature_cache: dict[Path, VoiceFeature] = {}
    paths_by_spell: dict[str, list[Path]] = {}
    for spell in spellbook.spells:
        paths = voice_sample_abs_paths(spellbook, spell)
        paths_by_spell[spell.id] = paths
        for p in paths:
            if p not in feature_cache and p.exists():
                feature_cache[p] = backend.extract_path(p, config)

    results: list[LooResult] = []
    for spell in spellbook.spells:
        if not spell.has_voice:
            continue
        for held_out in paths_by_spell[spell.id]:
            if held_out not in feature_cache:
                continue
            ranking = _rank_with_holdout(
                feature_cache[held_out],
                spellbook,
                spell.id,
                held_out,
                config,
                feature_cache,
                backend,
            )
            if not ranking:
                continue
            best = ranking[0]
            second = ranking[1] if len(ranking) > 1 else None
            intra_baseline = (
                best.intra_class_median
                if best.intra_class_median is not None
                else config.untrained_distance_fallback
            )
            intra_ratio = best.aggregate_distance / max(intra_baseline, 1e-9)
            margin_ratio: float | None
            if second is None or second.aggregate_distance <= 0.0:
                margin_ratio = None
            else:
                margin_ratio = (
                    second.aggregate_distance - best.aggregate_distance
                ) / second.aggregate_distance
            results.append(
                LooResult(
                    spell_id=spell.id,
                    spell_name=spell.name,
                    sample_path=held_out,
                    best_spell_id=best.spell_id,
                    best_spell_name=best.name,
                    best_distance=best.aggregate_distance,
                    second_distance=(
                        second.aggregate_distance if second is not None else None
                    ),
                    intra_ratio=intra_ratio,
                    margin_ratio=margin_ratio,
                    correct=best.spell_id == spell.id,
                )
            )
    return results


def _rank_with_holdout(
    query: VoiceFeature,
    spellbook: Spellbook,
    holdout_spell_id: str,
    holdout_path: Path,
    config: VoiceRecognitionConfig,
    feature_cache: dict[Path, VoiceFeature],
    backend: VoiceTemplateBackend,
) -> list[SpellRanking]:
    rankings: list[SpellRanking] = []
    for spell in spellbook.spells:
        if not spell.enabled or not spell.has_voice:
            continue
        sample_paths = [
            p
            for p in voice_sample_abs_paths(spellbook, spell)
            if not (spell.id == holdout_spell_id and p == holdout_path)
        ]
        if not sample_paths:
            continue
        per_sample = _distances_to_samples(
            query, sample_paths, config, feature_cache, backend
        )
        if not per_sample:
            continue
        # Recompute intra_class_median without the held-out sample, so the gate
        # reflects what the spellbook *would* know under genuine leave-one-out.
        if spell.id == holdout_spell_id:
            remaining_features = [
                feature_cache[p] for p in sample_paths if p in feature_cache
            ]
            intra = compute_intra_class_median(remaining_features, backend)
        else:
            intra = spell.intra_class_median
        rankings.append(
            SpellRanking(
                spell_id=spell.id,
                name=spell.name,
                aggregate_distance=backend.aggregate(per_sample),
                per_sample_distances=tuple(per_sample),
                intra_class_median=intra,
            )
        )
    rankings.sort(key=lambda r: r.aggregate_distance)
    return rankings


def _distances_to_samples(
    query: VoiceFeature,
    sample_paths: list[Path],
    config: VoiceRecognitionConfig,
    feature_cache: dict[Path, VoiceFeature] | None,
    backend: VoiceTemplateBackend,
) -> list[float]:
    distances: list[float] = []
    for path in sample_paths:
        if not path.exists():
            LOGGER.warning("Sample missing on disk: %s", path)
            continue
        if feature_cache is not None and path in feature_cache:
            template = feature_cache[path]
        else:
            template = backend.extract_path(path, config)
            if feature_cache is not None:
                feature_cache[path] = template
        distances.append(backend.distance(query, template))
    return distances


def default_voice_backend() -> VoiceTemplateBackend:
    from .parakeet_ctc_backends import parakeet_ctc_forced_backend

    return parakeet_ctc_forced_backend()
