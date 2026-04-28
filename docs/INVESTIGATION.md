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
- A follow-up diagnostic spike adds `faster-whisper-nbest`, which compares CTranslate2 beam hypotheses instead of encoder frames. CTranslate2 exposes multiple decoded sequences with `num_hypotheses`; the spike treats those as a weighted bag of plausible phonetic/text interpretations for each sample.
- A second follow-up spike adds `parakeet-ctc-forced`, which uses the Parakeet 110M CTC ONNX model's own timing-invariant forward algorithm to score each query posteriorgram against token sequences decoded from training samples. This is a better conceptual fit than DTW over CTC posterior frames.

## Current Runtime Candidate

`parakeet-ctc-forced` with `entropora/parakeet-ctc-110m-int8` is the current runtime recognizer candidate. It uses Parakeet 110M CTC posteriorgrams through ONNX Runtime and scores each query using CTC forced-sequence likelihood against token sequences decoded from the user's samples.

Corrected held-out calibration results from `session_20260424_204205`:

- `faster-whisper-dtw`: `49/60` positive hits and `0/10` false accepts at margin `0.15`; best zero-false-accept point is `53/60` at margin `0.10`.
- `faster-whisper-nbest`: `25/60` positive hits and `0/10` false accepts at margin `0.20`; this remains weaker than frame-level Whisper DTW.
- `parakeet-ctc-forced`: `57/60` positive hits and `0/10` false accepts from margin `0.07` through `0.20`, using Parakeet 110M INT8 CTC posteriorgrams and CTC forced-sequence scoring.

The current release path is being switched to `parakeet-ctc-forced` because false negatives were the more problematic practical failure mode, and the Parakeet CTC scorer gives the best current held-out recognition curve.

Measured local tradeoff on the corrected benchmark:

- Model files: bundled `faster-whisper-tiny` is about `74.6 MiB`; Parakeet 110M INT8 cache is about `125.6 MiB`, a `+51.0 MiB` model delta.
- Peak RSS during `diagnose --backend all`: `faster-whisper-dtw` about `147 MiB`; `parakeet-ctc-forced` about `368 MiB`, a `+221 MiB` process-memory delta when both are loaded in one process.
- Feature extraction: `faster-whisper-dtw` about `11.4s`; `parakeet-ctc-forced` about `4.0s` on the same session.

The full research backend spike, including MFCC, WavLM, Wav2Vec2, OpenWakeWord, Transformers Whisper, and faster-whisper comparison code, is preserved in git at commit `87e579f` (`Add faster-whisper diagnostic backend`).

## Open Questions

- `faster-whisper-nbest` tests whether user-trained spells can be represented by overlapping beam hypotheses such as `alohomora`, `aloha mora`, or similar token sequences. This may help where Whisper already knows a stable near-text interpretation, but it may fail for common English phrases or for nonsense words that decode inconsistently.
- `parakeet-ctc-forced` originally appeared false-positive-bound because diagnostics evaluated old calibration-session negatives against new live spellbook entries such as `Sneed`. Diagnostics now filter each session to spells that appear in that session's positive labels. After that correction, Parakeet has the best current threshold curve.
- Parakeet/FastConformer research dependencies should remain optional until packaging is revisited. The CTC forced scorer is useful evidence that ASR-native scoring can outperform generic embedding distance for this few-shot spell-recognition task.

## UX Implications

- Fast delivery is currently the hardest positive variant. Calibration and training UI should explicitly ask for varied pacing rather than assuming normal clean samples are representative.
- Diagnostics should present failure modes by variant so a user can understand whether they need to rerecord examples, slow down, choose more distinct incantations, or loosen a threshold.
- A separate online classifier is deferred. The current evidence favors improving sample capture, diagnostics, and threshold calibration before adding training complexity.

## Milestone Status

Milestone 1 is considered successful: the project has a baseline recognizer, a stronger neural recognizer candidate, calibration capture, threshold diagnostics, ROC/performance plots, and variant-aware calibration data.

The next milestone should focus on the user-facing training and recognition UI, initially as a desktop 2D UI that can later be displayed in the VR overlay.
