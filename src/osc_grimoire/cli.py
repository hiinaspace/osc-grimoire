from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .audio_capture import PushToTalkRecorder
from .calibration import (
    CalibrationExample,
    CalibrationReport,
    ThresholdSweepResult,
    diagnose_calibration_session,
    latest_calibration_session,
    write_calibration_metadata,
)
from .config import AppConfig, VoiceRecognitionConfig
from .paths import default_data_dir
from .spellbook import (
    Spell,
    Spellbook,
    add_voice_sample,
    create_spell,
    delete_spell,
    find_spell_by_name,
    load_spellbook,
    next_voice_sample_path,
    save_spellbook,
)
from .voice_recognizer import (
    MFCC_DTW_BACKEND,
    SpellRanking,
    VoiceTemplateBackend,
    decide,
    leave_one_out_eval,
    rank_spells,
    recompute_all,
    recompute_spell_voice_stats,
)


@dataclass(frozen=True)
class CalibrationPrompt:
    id: str | None
    name: str
    count: int
    prompt: str | None = None


LOGGER = logging.getLogger(__name__)
WHISPER_DTW_RELATIVE_MARGIN_MIN = 0.15


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    if args.hotkey:
        config = _replace_hotkey(config, args.hotkey)
    if args.device is not None:
        config = _replace_device(config, args.device)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    handler = _COMMANDS[args.command]
    return handler(args, config, data_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osc-grimoire",
        description="Voice-spell recognition CLI (Milestone 1).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the platformdirs data directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--hotkey",
        default=None,
        help="Push-to-talk key for recording (default: space).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Audio input device (index or name substring).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show data dir, audio devices, and spellbook summary.")

    p_train = sub.add_parser("train", help="Train a spell by recording samples.")
    p_train.add_argument("name", help="Spell name (will be created if new).")
    p_train.add_argument(
        "--samples", type=int, default=5, help="Samples to record (default 5)."
    )

    p_add = sub.add_parser("add-sample", help="Record one more sample for a spell.")
    p_add.add_argument("name")

    sub.add_parser("list", help="List spells and sample counts.")

    p_del = sub.add_parser("delete", help="Delete a spell and its sample files.")
    p_del.add_argument("name")
    p_del.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt."
    )

    sub.add_parser(
        "recognize",
        help="Loop: hold hotkey, speak, release; print the ranked match.",
    )

    sub.add_parser(
        "test",
        help="Leave-one-out evaluation across the spellbook.",
    )

    sub.add_parser(
        "recompute",
        help="Recompute per-spell intra-class median for every spell.",
    )

    p_neg = sub.add_parser(
        "record-negatives",
        help="Record gibberish/non-spell clips to a directory for rejection tests.",
    )
    p_neg.add_argument(
        "--out",
        default="tests/fixtures/voice_negatives",
        help="Output directory (default: tests/fixtures/voice_negatives).",
    )
    p_neg.add_argument(
        "--prefix",
        default="neg",
        help="Filename prefix for saved clips (default: neg).",
    )
    p_neg.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many clips to record (default: 10).",
    )

    p_cal = sub.add_parser(
        "calibrate",
        help="Record labeled attempts and negatives, then recommend thresholds.",
    )
    p_cal.add_argument(
        "--samples-per-spell",
        type=int,
        default=3,
        help="Positive attempts to record per spell (default: 3).",
    )
    p_cal.add_argument(
        "--negatives",
        type=int,
        default=10,
        help="Negative clips to record (default: 10).",
    )
    p_cal.add_argument(
        "--variant-plan",
        default=None,
        help=(
            "Positive prompt variants, e.g. standard or "
            "clean=5,quiet=5,slow=5,fast=5. Overrides --samples-per-spell."
        ),
    )

    p_diag = sub.add_parser(
        "diagnose",
        help="Analyze the latest calibration session, or one passed via --session.",
    )
    p_diag.add_argument(
        "--session",
        default=None,
        help="Calibration session directory (default: latest under data-dir).",
    )
    p_diag.add_argument(
        "--backend",
        default="mfcc-dtw",
        choices=[
            "mfcc-dtw",
            "wavlm-dtw",
            "wavlm-mean",
            "conformer-dtw",
            "conformer-mean",
            "w2vbert-dtw",
            "w2vbert-mean",
            "whisper-dtw",
            "whisper-mean",
            "oww-dtw",
            "oww-mean",
            "all",
        ],
        help="Recognizer backend to evaluate (default: mfcc-dtw).",
    )
    p_diag.add_argument(
        "--embedding-model",
        default=None,
        help="Hugging Face model for neural embedding backends.",
    )
    p_diag.add_argument(
        "--plot-dir",
        default=None,
        help="Write ROC and performance plots to this directory.",
    )

    return parser


def _replace_hotkey(config: AppConfig, hotkey: str) -> AppConfig:
    return replace(config, hotkey=hotkey)


def _replace_device(config: AppConfig, device: str) -> AppConfig:
    parsed: int | str
    try:
        parsed = int(device)
    except ValueError:
        parsed = device
    return replace(config, audio=replace(config.audio, input_device=parsed))


def _cmd_info(_args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    print(f"data dir: {data_dir}")
    print(f"hotkey  : {config.hotkey}")
    print(f"audio   : {config.audio}")
    print()
    hostapis = sd.query_hostapis()
    print("input devices:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = "*" if idx == sd.default.device[0] else " "
            host = hostapis[dev["hostapi"]]["name"]
            print(
                f"  {marker} [{idx:>3}] {dev['name']} "
                f"({dev['max_input_channels']} ch, {host})"
            )
    print()

    spellbook = load_spellbook(data_dir)
    print(f"spellbook: {len(spellbook.spells)} spell(s)")
    for spell in spellbook.spells:
        print(
            f"  - {spell.name} [{spell.id[:8]}] "
            f"voice={len(spell.voice_samples)} gesture={len(spell.gesture_samples)}"
        )
    return 0


def _cmd_train(args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    spell = find_spell_by_name(spellbook, args.name)
    if spell is None:
        spellbook, spell = create_spell(spellbook, args.name)
        print(f"Created new spell {spell.name!r} ({spell.id[:8]}).")
    else:
        print(
            f"Adding to existing spell {spell.name!r} "
            f"(currently {len(spell.voice_samples)} voice samples)."
        )

    print(
        f"Recording {args.samples} sample(s). Hold '{config.hotkey}' to record, "
        "release to stop."
    )

    spellbook = _record_samples(spellbook, spell, args.samples, config)
    spellbook = _recompute_and_report(spellbook, spell.id, config)
    save_spellbook(spellbook)

    fresh = find_spell_by_name(spellbook, args.name)
    assert fresh is not None
    print(f"Done. {fresh.name!r} now has {len(fresh.voice_samples)} voice sample(s).")
    return 0


def _cmd_add_sample(args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    spell = find_spell_by_name(spellbook, args.name)
    if spell is None:
        print(f"No spell named {args.name!r}.", file=sys.stderr)
        return 1

    print(f"Hold '{config.hotkey}' to record one sample for {spell.name!r}.")
    spellbook = _record_samples(spellbook, spell, 1, config)
    spellbook = _recompute_and_report(spellbook, spell.id, config)
    save_spellbook(spellbook)

    fresh = find_spell_by_name(spellbook, args.name)
    assert fresh is not None
    print(f"{fresh.name!r} now has {len(fresh.voice_samples)} voice sample(s).")
    return 0


def _recompute_and_report(
    spellbook: Spellbook, spell_id: str, config: AppConfig
) -> Spellbook:
    spell = next((s for s in spellbook.spells if s.id == spell_id), None)
    if spell is None:
        return spellbook
    spellbook = recompute_spell_voice_stats(spellbook, spell, config.voice)
    fresh = next((s for s in spellbook.spells if s.id == spell_id), None)
    if fresh is not None:
        if fresh.intra_class_median is not None:
            print(
                f"  intra-class median DTW for {fresh.name!r}: "
                f"{fresh.intra_class_median:.1f} (over "
                f"{len(fresh.voice_samples)} sample(s))"
            )
        else:
            print(
                f"  intra-class median for {fresh.name!r}: not computable "
                f"(need at least 2 samples)"
            )
    return spellbook


def _cmd_list(_args: argparse.Namespace, _config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    if not spellbook.spells:
        print("(no spells)")
        return 0
    name_w = max(len(s.name) for s in spellbook.spells)
    for spell in spellbook.spells:
        intra = (
            f"{spell.intra_class_median:7.1f}"
            if spell.intra_class_median is not None
            else "    n/a"
        )
        print(
            f"{spell.name:<{name_w}}  [{spell.id[:8]}]  "
            f"voice={len(spell.voice_samples):>2}  "
            f"gesture={len(spell.gesture_samples):>2}  "
            f"intra={intra}  "
            f"{'enabled' if spell.enabled else 'disabled'}"
        )
    return 0


def _cmd_delete(args: argparse.Namespace, _config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    spell = find_spell_by_name(spellbook, args.name)
    if spell is None:
        print(f"No spell named {args.name!r}.", file=sys.stderr)
        return 1
    if not args.yes:
        ans = input(
            f"Delete {spell.name!r} ({len(spell.voice_samples)} sample(s))? [y/N] "
        )
        if ans.strip().lower() != "y":
            print("Aborted.")
            return 0

    for rel in spell.voice_samples:
        path = data_dir / rel
        if path.exists():
            path.unlink()
    samples_root = data_dir / "samples" / f"spell_{spell.id}"
    if samples_root.exists():
        for f in samples_root.iterdir():
            f.unlink()
        samples_root.rmdir()

    spellbook = delete_spell(spellbook, spell.id)
    save_spellbook(spellbook)
    print(f"Deleted {spell.name!r}.")
    return 0


def _cmd_recognize(_args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    if not spellbook.spells:
        print("Spellbook is empty. Train some spells first.", file=sys.stderr)
        return 1

    missing_intra = [
        s.name
        for s in spellbook.spells
        if s.has_voice and s.voice_samples and s.intra_class_median is None
    ]
    if missing_intra:
        print(
            "  WARNING: spells without intra-class median: "
            f"{', '.join(missing_intra)}. Run `osc-grimoire recompute` first.",
            file=sys.stderr,
        )

    print(f"Hold '{config.hotkey}' and speak. Release to classify. Ctrl-C to quit.")
    feature_cache: dict[Path, np.ndarray] = {}

    with PushToTalkRecorder(
        config.audio, hotkey=config.hotkey, on_state_change=_print_recording_state
    ) as recorder:
        try:
            while True:
                audio = recorder.record_one()
                if audio.size == 0:
                    print("(no audio captured)")
                    continue
                duration = audio.size / config.audio.sample_rate
                print(f"  captured {duration:.2f}s of audio")
                query = MFCC_DTW_BACKEND.extract_array(
                    audio, config.voice, config.audio.sample_rate
                )
                ranking = rank_spells(
                    query,
                    spellbook,
                    config.voice,
                    feature_cache,
                    backend=MFCC_DTW_BACKEND,
                )
                if not ranking:
                    print("  no rankable spells")
                    continue
                _print_ranking(ranking)
                decision = decide(ranking, config.voice)
                verdict = "ACCEPTED" if decision.accepted else "rejected"
                _print_decision(decision, verdict)
                print()
        except KeyboardInterrupt:
            print()
            return 0


def _print_decision(decision, verdict: str) -> None:
    intra = (
        f"{decision.intra_ratio:.2f}/{decision.intra_ratio_max:.2f}"
        if decision.intra_ratio is not None
        else "n/a"
    )
    margin = (
        f"{decision.margin_ratio:.2f}/{decision.margin_ratio_min:.2f}"
        if decision.margin_ratio is not None
        else "n/a"
    )
    print(
        f"  decision: {verdict}  "
        f"intra_ratio={intra}  margin_ratio={margin}  ({decision.reason})"
    )


def _cmd_test(_args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    if not spellbook.spells:
        print("Spellbook is empty. Train some spells first.", file=sys.stderr)
        return 1

    results = leave_one_out_eval(spellbook, config.voice)
    if not results:
        print("No samples available for evaluation.")
        return 0

    correct = sum(1 for r in results if r.correct)
    total = len(results)
    print(f"Leave-one-out: {correct}/{total} correct ({100 * correct / total:.1f}%)")
    print()
    name_w = max(len(r.spell_name) for r in results)
    for r in results:
        marker = "OK" if r.correct else "MISS"
        sample = r.sample_path.name
        intra_repr = f"{r.intra_ratio:5.2f}" if r.intra_ratio is not None else "  n/a"
        margin_repr = (
            f"{r.margin_ratio:5.2f}" if r.margin_ratio is not None else "  n/a"
        )
        print(
            f"  [{marker:4}] {r.spell_name:<{name_w}}  {sample}  "
            f"best={r.best_spell_name:<{name_w}} "
            f"d={r.best_distance:7.2f}  "
            f"intra_ratio={intra_repr}  margin_ratio={margin_repr}"
        )

    print()
    confusions: Counter[tuple[str, str]] = Counter()
    for r in results:
        if not r.correct:
            confusions[(r.spell_name, r.best_spell_name)] += 1
    if confusions:
        print("Top confusions (true -> predicted, count):")
        for (true, pred), count in confusions.most_common():
            print(f"  {true} -> {pred}: {count}")
    return 0


def _cmd_recompute(_args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    if not spellbook.spells:
        print("(no spells)")
        return 0
    spellbook = recompute_all(spellbook, config.voice)
    save_spellbook(spellbook)
    for spell in spellbook.spells:
        intra = (
            f"{spell.intra_class_median:7.1f}"
            if spell.intra_class_median is not None
            else "    n/a"
        )
        print(f"  {spell.name}: intra_class_median={intra}")
    return 0


def _cmd_record_negatives(
    args: argparse.Namespace, config: AppConfig, _data_dir: Path
) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Hold '{config.hotkey}' and say things that should NOT match any spell. "
        f"Recording {args.count} clip(s) to {out_dir}."
    )
    n_existing = len(list(out_dir.glob(f"{args.prefix}_*.wav")))
    with PushToTalkRecorder(
        config.audio, hotkey=config.hotkey, on_state_change=_print_recording_state
    ) as recorder:
        for i in range(args.count):
            print(f"\nNegative {i + 1}/{args.count}: ready when you are…")
            audio = recorder.record_one()
            if audio.size == 0:
                print("  (no audio captured; skipping)")
                continue
            duration = audio.size / config.audio.sample_rate
            n = n_existing + i + 1
            path = out_dir / f"{args.prefix}_{n:03d}.wav"
            sf.write(str(path), audio, config.audio.sample_rate)
            print(f"  saved {path} ({duration:.2f}s)")
    return 0


def _cmd_calibrate(args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    spells = [s for s in spellbook.spells if s.enabled and s.has_voice]
    if not spells:
        print("No enabled voice spells. Train some spells first.", file=sys.stderr)
        return 1

    spellbook = recompute_all(spellbook, config.voice)
    save_spellbook(spellbook)
    spells = [s for s in spellbook.spells if s.enabled and s.has_voice]

    session_dir = (
        data_dir / "calibration" / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    examples: list[CalibrationExample] = []
    prompt_plan = _calibration_prompt_plan(args.variant_plan, args.samples_per_spell)
    print(f"Calibration session: {session_dir}")
    print(
        "This records held-out attempts. They are not added as training samples "
        "unless you choose to do that later."
    )
    if args.variant_plan:
        summary = ", ".join(f"{p.name} x{p.count}" for p in prompt_plan)
        print(f"Positive prompt variants: {summary}")

    with PushToTalkRecorder(
        config.audio, hotkey=config.hotkey, on_state_change=_print_recording_state
    ) as recorder:
        for spell in spells:
            spell_dir = session_dir / "positives" / f"{spell.id}_{spell.name}"
            spell_dir.mkdir(parents=True, exist_ok=True)
            for prompt in prompt_plan:
                variant_dir = spell_dir / prompt.id if prompt.id else spell_dir
                total = prompt.count
                for i in range(total):
                    style = f" Style: {prompt.prompt}" if prompt.prompt else ""
                    print(
                        f"\nSay {spell.name!r} ({prompt.name} {i + 1}/{total})."
                        f"{style} Hold '{config.hotkey}'."
                    )
                    audio = recorder.record_one()
                    example = _save_calibration_clip(
                        audio,
                        config,
                        variant_dir / f"attempt_{i + 1:03d}.wav",
                        kind="positive",
                        expected_spell_id=spell.id,
                        expected_spell_name=spell.name,
                        variant_id=prompt.id,
                        variant_name=prompt.name,
                        prompt=prompt.prompt,
                    )
                    if example is not None:
                        examples.append(example)

        negative_dir = session_dir / "negatives"
        negative_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.negatives):
            print(
                f"\nSay something that should NOT cast "
                f"({i + 1}/{args.negatives}). Hold '{config.hotkey}'."
            )
            audio = recorder.record_one()
            example = _save_calibration_clip(
                audio,
                config,
                negative_dir / f"negative_{i + 1:03d}.wav",
                kind="negative",
                expected_spell_id=None,
                expected_spell_name=None,
                variant_id=None,
                variant_name=None,
                prompt=None,
            )
            if example is not None:
                examples.append(example)

    write_calibration_metadata(session_dir, examples)
    report = diagnose_calibration_session(
        session_dir, spellbook, config.voice, MFCC_DTW_BACKEND
    )
    _print_calibration_report(report)
    return 0


def _save_calibration_clip(
    audio: np.ndarray,
    config: AppConfig,
    path: Path,
    *,
    kind: str,
    expected_spell_id: str | None,
    expected_spell_name: str | None,
    variant_id: str | None = None,
    variant_name: str | None = None,
    prompt: str | None = None,
) -> CalibrationExample | None:
    if audio.size == 0:
        print("  (no audio captured; skipping)")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, config.audio.sample_rate)
    duration = audio.size / config.audio.sample_rate
    print(f"  saved {path} ({duration:.2f}s)")
    return CalibrationExample(
        path=path,
        kind=kind,
        expected_spell_id=expected_spell_id,
        expected_spell_name=expected_spell_name,
        variant_id=variant_id,
        variant_name=variant_name,
        prompt=prompt,
    )


def _calibration_prompt_plan(
    variant_plan: str | None, samples_per_spell: int
) -> list[CalibrationPrompt]:
    if variant_plan is None:
        return [CalibrationPrompt(id=None, name="attempt", count=samples_per_spell)]
    plan = variant_plan.strip()
    if not plan:
        raise RuntimeError("--variant-plan cannot be empty")
    if plan.lower() == "standard":
        return [
            CalibrationPrompt(
                id="clean",
                name="clean",
                count=5,
                prompt="Use your normal spell voice.",
            ),
            CalibrationPrompt(
                id="quiet",
                name="quiet",
                count=5,
                prompt="Say it clearly but quieter than normal.",
            ),
            CalibrationPrompt(
                id="slow",
                name="slow",
                count=5,
                prompt="Say it clearly and slower than normal.",
            ),
            CalibrationPrompt(
                id="fast",
                name="fast",
                count=5,
                prompt="Say it clearly and faster than normal.",
            ),
        ]
    prompts: list[CalibrationPrompt] = []
    for item in plan.split(","):
        name, count = _parse_variant_plan_item(item)
        prompts.append(
            CalibrationPrompt(
                id=_slugify_variant(name),
                name=name,
                count=count,
                prompt=f"Use the {name} variation.",
            )
        )
    return prompts


def _parse_variant_plan_item(item: str) -> tuple[str, int]:
    if "=" not in item:
        raise RuntimeError(
            "Variant plan entries must look like name=count, e.g. clean=5,quiet=5."
        )
    raw_name, raw_count = item.split("=", 1)
    name = raw_name.strip()
    if not name:
        raise RuntimeError("Variant plan names cannot be empty.")
    try:
        count = int(raw_count.strip())
    except ValueError as exc:
        raise RuntimeError(f"Variant count for {name!r} must be an integer.") from exc
    if count <= 0:
        raise RuntimeError(f"Variant count for {name!r} must be positive.")
    return name, count


def _slugify_variant(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "variant"


def _cmd_diagnose(args: argparse.Namespace, config: AppConfig, data_dir: Path) -> int:
    spellbook = load_spellbook(data_dir)
    if not spellbook.spells:
        print("Spellbook is empty. Train some spells first.", file=sys.stderr)
        return 1
    session_dir = (
        Path(args.session) if args.session else latest_calibration_session(data_dir)
    )
    if session_dir is None:
        print("No calibration session found. Run `osc-grimoire calibrate` first.")
        return 1

    try:
        backends = _resolve_diagnose_backends(args.backend, args.embedding_model)
        reports = [
            diagnose_calibration_session(
                session_dir,
                spellbook,
                _diagnose_config_for_backend(config.voice, backend),
                backend,
            )
            for backend in backends
        ]
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if len(reports) == 1:
        _print_calibration_report(reports[0])
    else:
        _print_calibration_comparison(reports)
    if args.plot_dir:
        try:
            from .diagnostic_plots import write_diagnostic_plots

            paths = write_diagnostic_plots(reports, Path(args.plot_dir))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print()
        print("Wrote diagnostic plots:")
        for path in paths:
            print(f"  {path}")
    return 0


def _resolve_diagnose_backends(
    backend_name: str, embedding_model: str | None
) -> list[VoiceTemplateBackend]:
    if backend_name == "mfcc-dtw":
        return [MFCC_DTW_BACKEND]

    if backend_name in {
        "wavlm-dtw",
        "wavlm-mean",
        "conformer-dtw",
        "conformer-mean",
        "w2vbert-dtw",
        "w2vbert-mean",
        "whisper-dtw",
        "whisper-mean",
        "oww-dtw",
        "oww-mean",
        "all",
    }:
        if backend_name in {"oww-dtw", "oww-mean"}:
            try:
                from .openwakeword_backends import (
                    openwakeword_dtw_backend,
                    openwakeword_mean_backend,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "OpenWakeWord backend module could not be imported. "
                    "Run `uv sync --group oww`, then retry."
                ) from exc

            if backend_name == "oww-dtw":
                return [openwakeword_dtw_backend()]
            return [openwakeword_mean_backend()]

        try:
            from .voice_embedding_backends import (
                DEFAULT_CONFORMER_MODEL,
                DEFAULT_EMBEDDING_MODEL,
                DEFAULT_WAV2VEC2_BERT_MODEL,
                DEFAULT_WHISPER_MODEL,
                MissingEmbeddingDependenciesError,
                conformer_dtw_backend,
                conformer_mean_backend,
                missing_embedding_dependencies_message,
                wav2vec2_bert_dtw_backend,
                wav2vec2_bert_mean_backend,
                wavlm_dtw_backend,
                wavlm_mean_backend,
                whisper_dtw_backend,
                whisper_mean_backend,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Embedding backend module could not be imported. "
                "Run `uv sync --group ml`, then retry."
            ) from exc

        if backend_name == "wavlm-dtw":
            model = embedding_model or DEFAULT_EMBEDDING_MODEL
            return [wavlm_dtw_backend(model)]
        if backend_name == "wavlm-mean":
            model = embedding_model or DEFAULT_EMBEDDING_MODEL
            return [wavlm_mean_backend(model)]
        if backend_name == "conformer-dtw":
            model = embedding_model or DEFAULT_CONFORMER_MODEL
            return [conformer_dtw_backend(model)]
        if backend_name == "conformer-mean":
            model = embedding_model or DEFAULT_CONFORMER_MODEL
            return [conformer_mean_backend(model)]
        if backend_name == "w2vbert-dtw":
            model = embedding_model or DEFAULT_WAV2VEC2_BERT_MODEL
            return [wav2vec2_bert_dtw_backend(model)]
        if backend_name == "w2vbert-mean":
            model = embedding_model or DEFAULT_WAV2VEC2_BERT_MODEL
            return [wav2vec2_bert_mean_backend(model)]
        if backend_name == "whisper-dtw":
            model = embedding_model or DEFAULT_WHISPER_MODEL
            return [whisper_dtw_backend(model)]
        if backend_name == "whisper-mean":
            model = embedding_model or DEFAULT_WHISPER_MODEL
            return [whisper_mean_backend(model)]
        try:
            from .openwakeword_backends import (
                openwakeword_dtw_backend,
                openwakeword_mean_backend,
            )

            return [
                MFCC_DTW_BACKEND,
                wavlm_dtw_backend(DEFAULT_EMBEDDING_MODEL),
                wavlm_mean_backend(DEFAULT_EMBEDDING_MODEL),
                conformer_dtw_backend(DEFAULT_CONFORMER_MODEL),
                conformer_mean_backend(DEFAULT_CONFORMER_MODEL),
                wav2vec2_bert_dtw_backend(DEFAULT_WAV2VEC2_BERT_MODEL),
                wav2vec2_bert_mean_backend(DEFAULT_WAV2VEC2_BERT_MODEL),
                whisper_dtw_backend(DEFAULT_WHISPER_MODEL),
                whisper_mean_backend(DEFAULT_WHISPER_MODEL),
                openwakeword_dtw_backend(),
                openwakeword_mean_backend(),
            ]
        except MissingEmbeddingDependenciesError as exc:
            raise RuntimeError(missing_embedding_dependencies_message()) from exc

    raise RuntimeError(f"Unknown backend {backend_name!r}")


def _diagnose_config_for_backend(
    config: VoiceRecognitionConfig, backend: VoiceTemplateBackend
) -> VoiceRecognitionConfig:
    if backend.name.startswith("whisper-dtw:"):
        return replace(config, relative_margin_min=WHISPER_DTW_RELATIVE_MARGIN_MIN)
    return config


def _print_calibration_report(report: CalibrationReport) -> None:
    print()
    print(f"Diagnosis for {report.session_dir}")
    print(
        f"Backend: {report.backend_name}  "
        f"feature extraction: {report.extraction_seconds:.2f}s  "
        f"rss: {_rss_repr(report)}"
    )
    positives = [d for d in report.examples if d.example.kind == "positive"]
    negatives = [d for d in report.examples if d.example.kind == "negative"]
    positive_hits = sum(1 for d in positives if d.accepted and d.correct)
    positive_wrong = sum(1 for d in positives if d.accepted and not d.correct)
    negative_false_accepts = sum(1 for d in negatives if d.accepted)
    print(
        f"Current thresholds: positives accepted {positive_hits}/{len(positives)}, "
        f"wrong casts {positive_wrong}, negative false accepts "
        f"{negative_false_accepts}/{len(negatives)}"
    )
    _print_variant_breakdown(positives)

    problem_examples = [
        d
        for d in report.examples
        if (d.example.kind == "positive" and not (d.accepted and d.correct))
        or (d.example.kind == "negative" and d.accepted)
    ]
    if problem_examples:
        print()
        print("Examples needing attention:")
        for d in problem_examples[:12]:
            expected = d.example.expected_spell_name or "(negative)"
            best = d.best_spell_name or "(none)"
            intra = f"{d.intra_ratio:.2f}" if d.intra_ratio is not None else "n/a"
            margin = f"{d.margin_ratio:.2f}" if d.margin_ratio is not None else "n/a"
            variant = (
                f" variant={d.example.variant_name},"
                if d.example.variant_name is not None
                else ""
            )
            print(
                f"  {d.example.path.name}:{variant} expected={expected}, best={best}, "
                f"accepted={d.accepted}, intra={intra}, margin={margin}"
            )

    print()
    print("Margin threshold sweep:")
    for row in report.sweep:
        print(
            f"  margin>={row.margin_min:>4.2f}: "
            f"positive hits {row.positive_correct:>2}/{row.positive_total:<2}  "
            f"wrong casts {row.positive_wrong:>2}  "
            f"false accepts {row.negative_accepted:>2}/{row.negative_total:<2}"
            f"{_variant_sweep_repr(row)}"
        )

    print()
    if report.recommended_margin_min is None:
        print(
            "No margin threshold cleanly accepted positives while rejecting all "
            "negatives. Add more distinct samples, delete outliers, or try a "
            "different recognizer."
        )
    else:
        print(
            "Recommended relative_margin_min: "
            f"{report.recommended_margin_min:.2f} "
            "(best zero-false-accept point in this session)."
        )


def _print_calibration_comparison(reports: list[CalibrationReport]) -> None:
    print()
    print(f"Diagnosis for {reports[0].session_dir}")
    print("Backend comparison:")
    for report in reports:
        positives = [d for d in report.examples if d.example.kind == "positive"]
        negatives = [d for d in report.examples if d.example.kind == "negative"]
        positive_hits = sum(1 for d in positives if d.accepted and d.correct)
        negative_false_accepts = sum(1 for d in negatives if d.accepted)
        recommended = (
            f"{report.recommended_margin_min:.2f}"
            if report.recommended_margin_min is not None
            else "none"
        )
        print(
            f"  {report.backend_name}: current hits "
            f"{positive_hits}/{len(positives)}, false accepts "
            f"{negative_false_accepts}/{len(negatives)}, "
            f"recommended margin {recommended}, "
            f"features {report.extraction_seconds:.2f}s, "
            f"rss {_rss_repr(report)}"
        )

    print()
    print("Margin threshold sweep:")
    margin_values = [row.margin_min for row in reports[0].sweep]
    for margin in margin_values:
        parts = []
        for report in reports:
            row = next(r for r in report.sweep if r.margin_min == margin)
            parts.append(
                f"{report.backend_name}: "
                f"{row.positive_correct}/{row.positive_total} hits, "
                f"{row.negative_accepted}/{row.negative_total} FA"
            )
        print(f"  margin>={margin:>4.2f}: " + " | ".join(parts))


def _variant_sweep_repr(row: ThresholdSweepResult) -> str:
    if not row.variants:
        return ""
    parts = [
        f"{v.variant_name} {v.positive_correct}/{v.positive_total}"
        + (f" ({v.positive_wrong} wrong)" if v.positive_wrong else "")
        for v in row.variants
    ]
    return "  variants: " + ", ".join(parts)


def _print_variant_breakdown(positives: list) -> None:
    variant_names = sorted(
        {d.example.variant_name for d in positives if d.example.variant_name}
    )
    if not variant_names:
        return
    print("Positive breakdown by variant:")
    for variant_name in variant_names:
        variant_examples = [
            d for d in positives if d.example.variant_name == variant_name
        ]
        hits = sum(1 for d in variant_examples if d.accepted and d.correct)
        wrong = sum(1 for d in variant_examples if d.accepted and not d.correct)
        print(
            f"  {variant_name}: {hits}/{len(variant_examples)} accepted"
            f" correct, wrong casts {wrong}"
        )


def _rss_repr(report: CalibrationReport) -> str:
    if report.peak_rss_mb is None:
        return "n/a"
    return f"{report.peak_rss_mb:.0f} MiB"


def _record_samples(
    spellbook: Spellbook, spell: Spell, count: int, config: AppConfig
) -> Spellbook:
    with PushToTalkRecorder(
        config.audio, hotkey=config.hotkey, on_state_change=_print_recording_state
    ) as recorder:
        for i in range(count):
            print(f"\nSample {i + 1}/{count}: ready when you are…")
            audio = recorder.record_one()
            if audio.size == 0:
                print("  (no audio captured; skipping)")
                continue
            duration = audio.size / config.audio.sample_rate
            current_spell = next(s for s in spellbook.spells if s.id == spell.id)
            wav_abs, wav_rel = next_voice_sample_path(spellbook, current_spell)
            sf.write(str(wav_abs), audio, config.audio.sample_rate)
            spellbook = add_voice_sample(spellbook, current_spell, wav_rel)
            print(f"  saved {wav_rel} ({duration:.2f}s)")
    return spellbook


def _print_recording_state(recording: bool) -> None:
    if recording:
        print("  [recording…]", end="\r", flush=True)
    else:
        print("  [released]   ", flush=True)


def _print_ranking(ranking: list[SpellRanking]) -> None:
    name_w = max(len(r.name) for r in ranking)
    for i, r in enumerate(ranking[:5]):
        marker = "*" if i == 0 else " "
        intra = (
            f"{r.intra_class_median:7.1f}"
            if r.intra_class_median is not None
            else "    n/a"
        )
        print(
            f"  {marker} {r.name:<{name_w}}  "
            f"d={r.aggregate_distance:7.2f}  intra_med={intra}  "
            f"per_sample=[{', '.join(f'{d:.2f}' for d in r.per_sample_distances)}]"
        )


_CommandHandler = Callable[[argparse.Namespace, AppConfig, Path], int]

_COMMANDS: dict[str, _CommandHandler] = {
    "info": _cmd_info,
    "train": _cmd_train,
    "add-sample": _cmd_add_sample,
    "list": _cmd_list,
    "delete": _cmd_delete,
    "recognize": _cmd_recognize,
    "test": _cmd_test,
    "recompute": _cmd_recompute,
    "record-negatives": _cmd_record_negatives,
    "calibrate": _cmd_calibrate,
    "diagnose": _cmd_diagnose,
}
