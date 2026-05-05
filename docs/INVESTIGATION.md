# Voice Recognition Investigation

This document records how OSC Grimoire arrived at its current local voice
recognizer and how spell matching works now.

## Current Matcher

Production voice recognition is text-first. A spell has a display name and zero
or more text incantations. The display name is used for UI and default OSC
parameter naming; incantations are the spoken phrases that can cast the spell.

On a voice attempt:

1. The app records while the voice trigger is held, then trims and resamples the
   audio to 16 kHz.
2. `entropora/parakeet-ctc-110m-int8` produces frame-by-frame CTC token log
   probabilities for the query audio.
3. Each enabled spell incantation is normalized, tokenized with Parakeet's
   `vocab.txt`, and scored against the query with the CTC forced-sequence forward
   algorithm.
4. The incantation distance is the negative average log probability per target
   token. Lower is better.
5. A spell's voice score is the lowest distance among its incantations. A spell
   with no incantations is skipped for voice matching.

The app accepts the best voice spell only if both gates pass:

- absolute gate: best incantation distance <= `voice_alias_distance_max`
- margin gate: `(second_distance - best_distance) / second_distance >= relative_margin_min`

The margin gate is skipped when only one spell has voice incantations. The `Voice`
strictness slider adjusts both the absolute distance gate and the relative margin
gate. The current defaults are intentionally lenient for responsiveness:
`voice_alias_distance_max = 7.0` and `relative_margin_min = 0.20`.

The UI also runs a small CTC prefix beam search over the same query posteriorgram.
Those top phrase hypotheses are displayed as pending incantations on the spell
page. Pending rows are scored with the same forced-sequence scorer, so a user can
see whether a heard phrase would match their pronunciation better before adding
it as a saved incantation.

Recorded audio examples are not part of production spell definitions. They remain
useful for diagnostics, calibration sessions, fixture tests, and recognizer/model
comparisons.

## Diagnostic Calibration

Calibration sessions contain positive and negative recorded clips. Diagnostics
load the current spellbook, filter to the spells represented in a session's
positive labels, and evaluate the same incantation matcher used by production.

Diagnostic output tracks:

- accepted spell
- best incantation source
- best distance
- relative margin
- top CTC phrase hypotheses
- false accepts, false rejects, and variant-level positive results

Threshold sweeps vary `relative_margin_min` while keeping the absolute distance
gate from the provided config. This keeps diagnostics aligned with the live
strictness model without reintroducing sample-template training.

## Historical Findings

The first milestone evaluated several recognizer families on recorded clips for
`lumos`, `flipendo`, `alohomora`, plus negative clips:

- MFCC+DTW was a useful baseline but did not separate short spell words well
  enough without admitting negatives.
- WavLM frame embeddings with DTW improved over MFCC+DTW but still left false
  accepts or false rejects on available calibration sessions.
- Mean-pooled embeddings were weaker than frame-level comparisons; timing and
  phonetic progression matter for short incantations.
- Wav2Vec2-BERT and large Wav2Vec2-Conformer models did not justify their memory
  and latency cost in the current harness.
- OpenWakeWord's shared `speech_embedding` ONNX extractor was fast and lightweight
  but performed worse than WavLM and Whisper on this vocabulary.
- Whisper encoder frame embeddings with DTW performed best among the early
  embedding approaches. `openai/whisper-tiny` matched or beat larger variants on
  the recorded sets while staying smaller and faster.
- A `faster-whisper-nbest` diagnostic spike compared CTranslate2 beam hypotheses
  as a weighted bag of plausible text interpretations. It was weaker than
  frame-level Whisper DTW on the tested data.
- The Parakeet CTC forced scorer gave the best held-out recognition curve and
  matched the current text-incantation product direction better than audio
  templates.

Corrected held-out calibration results from `session_20260424_204205`:

- `faster-whisper-dtw`: `49/60` positive hits and `0/10` false accepts at margin
  `0.15`; best zero-false-accept point was `53/60` at margin `0.10`.
- `faster-whisper-nbest`: `25/60` positive hits and `0/10` false accepts at
  margin `0.20`.
- `parakeet-ctc-forced`: `57/60` positive hits and `0/10` false accepts from
  margin `0.07` through `0.20`, using Parakeet 110M INT8 posteriorgrams and CTC
  forced-sequence scoring.

Measured local tradeoff on that benchmark:

- Model files: bundled `faster-whisper-tiny` was about `74.6 MiB`; Parakeet 110M
  INT8 cache was about `125.6 MiB`, a `+51.0 MiB` model delta.
- Peak RSS during `diagnose --backend all`: `faster-whisper-dtw` about `147 MiB`;
  `parakeet-ctc-forced` about `368 MiB`, a `+221 MiB` process-memory delta when
  both were loaded in one process.
- Feature extraction: `faster-whisper-dtw` about `11.4s`; `parakeet-ctc-forced`
  about `4.0s` on the same session.

The broader research backend spike, including MFCC, WavLM, Wav2Vec2,
OpenWakeWord, Transformers Whisper, and faster-whisper comparison code, is
preserved in git at commit `87e579f` (`Add faster-whisper diagnostic backend`).

## Remaining Questions

- Fast delivery remains a hard positive variant. Calibration fixtures should keep
  varied pacing so recognizer changes are tested against it.
- Nonsense words can tokenize and score well with Parakeet, but some spell names
  may be consistently heard as nearby ordinary words. The incantation UI handles
  this by letting users add those heard phrases as accepted spell incantations.
- The current app scores voice and gesture independently. A future combined
  voice+gesture confidence model might help similar spells, but the current
  explicit gates are easier to tune and debug.
