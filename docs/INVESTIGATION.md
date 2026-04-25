# Voice Recognition Investigation

This document records the first milestone investigation into local voice spell recognition for OSC Grimoire.

## Goal

Build a local recognizer that can distinguish a small user-trained spell vocabulary while still rejecting ordinary non-spell speech. The target UX is not speech-to-text; it is reliable intent detection for short incantation-like phrases.

## Methodology

- Recorded 5 training clips each for `lumos`, `flipendo`, and `alohomora`.
- Recorded negative clips that should not cast.
- Added held-out calibration sessions so positive attempts and negatives are evaluated without being added to the training spellbook.
- Evaluated threshold sweeps using false accepts and false rejects, with special attention to the zero-false-accept operating point.
- Added prompted calibration variants: clean, quiet, slow, and fast delivery.
- Compared MFCC+DTW, WavLM/HuBERT-style embeddings, Wav2Vec2-Conformer, Wav2Vec2-BERT, OpenWakeWord speech embeddings, and Whisper encoder embeddings.

## Findings

- MFCC+DTW is useful as a simple baseline but does not separate these short spell words well enough. It rejected many clear real incantations unless the margin threshold was loosened enough to also admit negatives.
- WavLM frame embeddings with DTW improved substantially over MFCC+DTW, but still left false accepts or false rejects on the available calibration sessions.
- Mean-pooled embeddings were generally weaker than frame-level DTW for this task. Timing and phonetic progression matter for short incantations.
- Wav2Vec2-BERT and large Wav2Vec2-Conformer models did not justify their memory and latency cost in the current harness.
- OpenWakeWord's shared `speech_embedding` ONNX feature extractor is fast and lightweight, but performed worse than WavLM and Whisper on this vocabulary. It remains useful as a deployment reference, not the leading recognizer.
- Whisper encoder frame embeddings with DTW performed best. `openai/whisper-tiny` matched or beat larger variants on the recorded sets while staying smaller and faster.

## Current Best Candidate

`whisper-dtw` with `openai/whisper-tiny` is the best milestone-one recognizer candidate.

Observed calibration results:

- Initial held-out session: `15/15` positive hits at `0/15` false accepts with `relative_margin_min=0.15`.
- Variant session with clean/quiet/slow/fast delivery: `53/60` positive hits at `0/10` false accepts with `relative_margin_min=0.15`.
- At the previous `0.20` margin, the variant session dropped to `44/60` positive hits, mostly from fast delivery.

The current candidate threshold for `whisper-dtw` is therefore `relative_margin_min=0.15`. MFCC+DTW should remain more conservative until replaced or separately calibrated.

## UX Implications

- Fast delivery is currently the hardest positive variant. Calibration and training UI should explicitly ask for varied pacing rather than assuming normal clean samples are representative.
- Diagnostics should present failure modes by variant so a user can understand whether they need to rerecord examples, slow down, choose more distinct incantations, or loosen a threshold.
- A separate online classifier is deferred. The current evidence favors improving sample capture, diagnostics, and threshold calibration before adding training complexity.

## Milestone Status

Milestone 1 is considered successful: the project has a baseline recognizer, a stronger neural recognizer candidate, calibration capture, threshold diagnostics, ROC/performance plots, and variant-aware calibration data.

The next milestone should focus on the user-facing training and recognition UI, initially as a desktop 2D UI that can later be displayed in the VR overlay.
