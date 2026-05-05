from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import VoiceRecognitionConfig
from .spellbook import Spellbook
from .voice_features import FloatArray

VoiceFeature = Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextHypothesis:
    text: str
    token_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class VoiceTemplateBackend:
    name: str
    extract_path: Callable[[Path, VoiceRecognitionConfig], VoiceFeature]
    extract_array: Callable[[FloatArray, VoiceRecognitionConfig, int], VoiceFeature]
    distance: Callable[[VoiceFeature, VoiceFeature], float]
    aggregate: Callable[[list[float]], float]
    tokenize_text: Callable[[str], tuple[int, ...]] | None = None
    distance_to_tokens: Callable[[VoiceFeature, tuple[int, ...]], float] | None = None
    text_hypotheses: (
        Callable[[VoiceFeature, int], tuple[TextHypothesis, ...]] | None
    ) = None


@dataclass(frozen=True)
class SpellRanking:
    spell_id: str
    name: str
    aggregate_distance: float
    alias: str
    token_ids: tuple[int, ...]
    alias_rankings: tuple[AliasRanking, ...] = ()


@dataclass(frozen=True)
class AliasRanking:
    alias: str
    distance: float
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    best_distance: float | None
    distance_max: float
    margin_ratio: float | None
    margin_ratio_min: float


def rank_spells(
    query: VoiceFeature,
    spellbook: Spellbook,
    config: VoiceRecognitionConfig,
    feature_cache: dict[Path, VoiceFeature] | None = None,
    backend: VoiceTemplateBackend | None = None,
    backend_stats: object | None = None,
) -> list[SpellRanking]:
    del config, feature_cache, backend_stats
    backend = backend or default_voice_backend()
    if backend.tokenize_text is None or backend.distance_to_tokens is None:
        raise RuntimeError(f"{backend.name} cannot score text incantations")

    rankings: list[SpellRanking] = []
    for spell in spellbook.spells:
        if not spell.enabled or not spell.has_voice:
            continue
        aliases: list[AliasRanking] = []
        for alias in spell.voice_aliases:
            try:
                token_ids = backend.tokenize_text(alias)
            except ValueError as exc:
                LOGGER.warning("Skipping invalid incantation %r: %s", alias, exc)
                continue
            distance = backend.distance_to_tokens(query, token_ids)
            aliases.append(
                AliasRanking(alias=alias, distance=distance, token_ids=token_ids)
            )
        aliases.sort(key=lambda row: row.distance)
        if not aliases:
            continue
        best = aliases[0]
        rankings.append(
            SpellRanking(
                spell_id=spell.id,
                name=spell.name,
                aggregate_distance=best.distance,
                alias=best.alias,
                token_ids=best.token_ids,
                alias_rankings=tuple(aliases),
            )
        )
    rankings.sort(key=lambda r: r.aggregate_distance)
    return rankings


def decide(
    rankings: list[SpellRanking],
    config: VoiceRecognitionConfig,
) -> Decision:
    if not rankings:
        return Decision(
            accepted=False,
            reason="no enabled spells with incantations",
            best_distance=None,
            distance_max=config.voice_alias_distance_max,
            margin_ratio=None,
            margin_ratio_min=config.relative_margin_min,
        )

    best = rankings[0]
    second = rankings[1] if len(rankings) > 1 else None
    margin_ratio: float | None
    if second is None or second.aggregate_distance <= 0.0:
        margin_ratio = None
    else:
        margin_ratio = (
            second.aggregate_distance - best.aggregate_distance
        ) / second.aggregate_distance

    if best.aggregate_distance > config.voice_alias_distance_max:
        return Decision(
            accepted=False,
            reason=(
                f"incantation distance {best.aggregate_distance:.2f} exceeds "
                f"{config.voice_alias_distance_max:.2f}"
            ),
            best_distance=best.aggregate_distance,
            distance_max=config.voice_alias_distance_max,
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
            best_distance=best.aggregate_distance,
            distance_max=config.voice_alias_distance_max,
            margin_ratio=margin_ratio,
            margin_ratio_min=config.relative_margin_min,
        )

    return Decision(
        accepted=True,
        reason="both gates pass",
        best_distance=best.aggregate_distance,
        distance_max=config.voice_alias_distance_max,
        margin_ratio=margin_ratio,
        margin_ratio_min=config.relative_margin_min,
    )


def text_hypotheses(
    query: VoiceFeature,
    backend: VoiceTemplateBackend | None = None,
    *,
    limit: int = 5,
) -> tuple[TextHypothesis, ...]:
    backend = backend or default_voice_backend()
    if backend.text_hypotheses is None:
        return ()
    return backend.text_hypotheses(query, limit)


def default_voice_backend() -> VoiceTemplateBackend:
    from .parakeet_ctc_backends import parakeet_ctc_forced_backend

    return parakeet_ctc_forced_backend()


def median_aggregate(distances: list[float]) -> float:
    return float(np.median(distances))
