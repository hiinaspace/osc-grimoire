# CHARTER.md — OSC Grimoire

## Project summary

OSC Grimoire is a local OpenVR overlay app for VRChat spell-casting gimmicks.

It lets a user train custom “spells” using one or both of:

- a hand/wand gesture traced in VR
- a spoken incantation recorded from the microphone

At runtime, the user holds a controller trigger, performs the gesture and/or says the incantation, releases the trigger, and the app matches the captured input against the trained spellbook. On a confident match, the app emits OSC parameters to VRChat, where avatars can use those parameters to drive animator states, particle effects, sounds, toggles, or other spell effects.

The intended design is not general speech-to-text and not always-listening wake-word detection. It is closer to “query-by-example” recognition: record a short clip or gesture, compare it to examples the user has trained, and choose the closest match if it is confidently close enough.

## Core goals

Build a Python-first prototype that can:

1. Display an OpenVR quad overlay attached to the user’s offhand, styled like a small spellbook UI.
2. Let the user create, edit, delete, and test spells.
3. Record gesture samples using the main-hand controller while the trigger is held.
4. Record voice samples from the microphone while the trigger is held.
5. Match a runtime gesture and/or voice clip against trained examples.
6. Emit VRChat OSC parameters for:
   - currently recognizing / casting
   - recognized spell ID
   - optional per-spell boolean pulses
   - optional confidence/debug values
7. Persist the spellbook locally in a human-inspectable format.
8. Support wand-tip offset calibration, so gesture traces can come from the avatar’s wand tip rather than the raw controller origin.

## Non-goals for MVP

Do not build a general-purpose STT system.

Do not attempt always-on wake-word detection.

Do not require cloud APIs.

Do not require training a neural network for MVP.

Do not solve multiplayer synchronization directly. The app only emits local OSC to VRChat; avatar/world logic handles the rest.

Do not build a full VR UI framework. The overlay UI can be simple and utilitarian.

Do not require support for non-SteamVR runtimes in the MVP. OpenVR/SteamVR is the target because VRChat overlay use is the practical requirement.

## Target user flow

### Training a spell

1. User opens the offhand spellbook overlay.
2. User presses “Add Spell.”
3. User enters or auto-generates a spell name.
4. User chooses which modalities to train:
   - gesture only
   - voice only
   - gesture + voice
5. For each sample:
   - UI prompts “hold main-hand trigger and perform the spell.”
   - while trigger is held:
     - gesture trace is captured from main-hand controller/wand-tip position
     - microphone audio is recorded, if voice training is enabled
     - app emits `currently_recognizing = true`
   - on trigger release:
     - trace and/or audio clip is saved as one training sample
6. User records several samples, likely 3–10.
7. App shows rough intra-class consistency metrics.
8. User assigns an OSC binding, such as:
   - `/avatar/parameters/osc-grimoire-spell-1`
   - `/avatar/parameters/Fireball`
   - `/avatar/parameters/SpellIndex`
9. Spell is saved.

### Recognizing a spell

1. User holds main-hand trigger.
2. App starts recording gesture/audio.
3. App emits a “currently recognizing” OSC boolean, e.g.

   ```text
   /avatar/parameters/osc-grimoire-casting = true
   ```

4. User traces the gesture and/or says the incantation.
5. User releases trigger.
6. App preprocesses the captured query.
7. App compares query against trained spell samples.
8. If the best match passes threshold and margin checks:

   * emit the spell’s OSC parameter
   * optionally emit a numeric spell index
   * optionally emit confidence/debug values
9. App resets `currently_recognizing` to false.

## Interaction model

### Controller roles

Default assumption:

* Offhand controller: spellbook anchor and UI interaction
* Main-hand controller: wand / casting hand

The offhand overlay is attached to the offhand controller as a quad overlay. It should be easy to glance at, but should not obstruct normal VRChat use.

The main hand is used to record gestures. The gesture point should be:

```text
gesture_point_world =
    main_hand_controller_pose_world
    * calibrated_controller_to_wand_tip_offset
```

For MVP, raw controller position can be used before wand-tip calibration exists.

### Wand-tip offset calibration

The app should support a simple calibration flow:

1. User equips avatar with wand.
2. User brings offhand controller close to the visible wand tip.
3. User holds a calibration button.
4. App computes the offset from main-hand controller pose to the offhand/world reference point.
5. App saves this as `controller_to_wand_tip_offset`.

A simple MVP version:

```text
controller_to_tip_world = offhand_position_world - main_hand_position_world
controller_to_tip_local = inverse(main_hand_rotation_world) * controller_to_tip_world
```

At runtime:

```text
tip_position_world =
    main_hand_position_world
    + main_hand_rotation_world * controller_to_tip_local
```

This ignores exact wand orientation and only calibrates a positional offset. That is probably sufficient for gesture paths.

## Recognition design

The system should treat recognition as nearest-neighbor matching with rejection.

For every recognition attempt, compute:

```text
best_spell
best_score
second_best_score
confidence_margin
```

Only trigger if:

```text
best_score <= absolute_threshold
and
second_best_score - best_score >= margin_threshold
```

For similarity systems where higher is better, invert this accordingly.

The margin check matters. It prevents accidental firing when the query is equally close to several spells.

## Gesture recognition MVP

Use a simple classical gesture recognizer first.

Candidate baseline:

* resample trace to fixed number of points
* normalize translation
* normalize scale
* optionally normalize rotation
* compare with a `$1`, Protractor, `$P`, or `$Q`-style recognizer
* keep multiple templates per spell

For 3D VR gestures, start with one of these approaches:

### Option A: project to a 2D casting plane

Infer or define a plane relative to the user/controller, then project the 3D wand-tip trace into 2D.

Possible planes:

* plane facing the headset
* plane fixed relative to the offhand spellbook
* plane fixed relative to the main hand at gesture start
* best-fit plane through the traced points

This makes `$Q`/point-cloud-style recognition easy.

### Option B: use 3D point-cloud matching

Normalize the 3D point sequence directly and compare with DTW or point-cloud distance.

This avoids projection ambiguity, but may be more sensitive to user stance and controller orientation.

### Recommended MVP

Start with projection to a 2D plane facing the headset or anchored at gesture start. Then use a `$Q`-style point-cloud recognizer. The spell gestures are likely symbolic glyphs, so 2D projection is a good first approximation.

### Gesture preprocessing

For each captured trace:

1. Remove samples with invalid poses.
2. Optionally smooth positions.
3. Resample to N points, e.g. 64.
4. Normalize translation to centroid.
5. Normalize scale to unit bounding box or unit path length.
6. Optionally normalize rotation.
7. Store normalized template.

Store the raw trace too for debugging and future recognizer improvements.

## Voice recognition MVP

Use template-based isolated-word / short-phrase matching.

Recommended baseline:

* record the clip while trigger is held
* trim leading/trailing silence lightly, if needed
* compute MFCC or log-mel features
* compare query feature sequence to training feature sequences using Dynamic Time Warping
* keep multiple templates per spell
* choose the spell whose examples have the best aggregate score

This is intentionally not STT. The word “flipendo” does not need to be spelled, tokenized, or decoded. It only needs to be acoustically similar to the user’s own training examples.

### Voice preprocessing

For each audio clip:

1. Convert to mono.
2. Resample to a fixed rate, probably 16 kHz.
3. Normalize volume conservatively.
4. Optionally trim silence at start/end.
5. Extract MFCCs, e.g. 13 coefficients plus deltas if useful.
6. Store:

   * original WAV/FLAC/OGG
   * extracted feature matrix
   * preprocessing metadata

### Voice matching

For each spell:

1. Compute DTW distance from query features to each training example.
2. Aggregate distances:

   * minimum distance: forgiving
   * median of k nearest examples: more stable
   * average after dropping worst outlier: good compromise
3. Compare across spells.
4. Trigger only if absolute threshold and margin threshold pass.

### Why this should work

The interaction is button-gated, speaker-dependent, and small-vocabulary. That removes most of the hard parts of wake-word/STT systems:

* no continuous listening
* no need to find utterance boundaries with VAD
* no need to recognize arbitrary language
* no need to support unknown speakers
* no need to decode fake words
* no need to distinguish speech from long ambient conversation

The main hard parts left are noise, inconsistent timing, and near-similar spells. MFCC + DTW is a reasonable baseline for those.

## Multimodal gesture + voice matching

A spell may have:

* only gesture templates
* only voice templates
* both gesture and voice templates

For a spell with both modalities, combine normalized scores.

Example:

```text
combined_score =
    gesture_weight * normalized_gesture_score
    + voice_weight * normalized_voice_score
```

Lower score means better match.

Each modality needs its own threshold and calibration because raw DTW scores are not directly comparable.

A safer first version:

```text
gesture_pass = gesture_score <= gesture_threshold
voice_pass = voice_score <= voice_threshold

if spell uses both:
    pass only if gesture_pass and voice_pass
elif spell uses gesture:
    pass if gesture_pass
elif spell uses voice:
    pass if voice_pass
```

This avoids premature score-fusion tuning.

Later, add a combined confidence model.

## OSC behavior

Default OSC host/port should target VRChat’s local OSC receiver.

Common default:

```text
host = 127.0.0.1
port = 9000
```

Configurable output examples:

```text
/avatar/parameters/osc-grimoire-casting        bool
/avatar/parameters/osc-grimoire-spell-1        bool pulse
/avatar/parameters/osc-grimoire-spell-2        bool pulse
/avatar/parameters/osc-grimoire-spell-index    int
/avatar/parameters/osc-grimoire-confidence     float
/avatar/parameters/Fireball                    bool pulse
```

Recommended default behavior:

1. On trigger press:

   ```text
   casting = true
   ```

2. On trigger release:

   ```text
   casting = false
   ```

3. On successful recognition:

   ```text
   spell-specific bool = true
   wait pulse_duration_ms
   spell-specific bool = false
   ```

4. Optionally also emit:

   ```text
   spell_index = N
   confidence = C
   ```

Pulse duration should be configurable, e.g. 100–500 ms.

The app should avoid permanently leaving booleans true after crashes. On startup/shutdown, emit reset values for known managed parameters where practical.

## VRChat avatar integration assumptions

The avatar can use the emitted OSC parameters to drive animator states.

Examples:

* `osc-grimoire-casting` turns on a wand trail or charging particle effect.
* `osc-grimoire-spell-1` triggers fireball animation.
* `osc-grimoire-spell-2` triggers levitation animation.
* `osc-grimoire-spell-index` selects from an animator blend tree or state machine.

The app should not require any specific avatar controller layout. It only provides OSC output.

## UI requirements

MVP UI can be simple.

Required screens:

### Main spellbook

* list of spells
* recognize/test mode
* add spell
* settings

### Spell editor

* spell name
* OSC parameter path
* modality flags:

  * gesture enabled
  * voice enabled
* sample list
* record new sample
* delete sample
* test recognition
* threshold controls
* last match diagnostics

### Calibration

* choose main/offhand roles
* calibrate wand-tip offset
* test current wand-tip position
* reset calibration

### Debug

* OpenVR pose status
* mic input level
* OSC connection/output log
* last gesture score table
* last voice score table
* last emitted OSC messages

## Data model

Use a local project directory, e.g.

```text
~/.config/osc-grimoire/
  config.json
  spellbook.json
  samples/
    spell_<uuid>/
      gesture_001.json
      gesture_002.json
      voice_001.wav
      voice_001.features.npy
      voice_002.wav
      voice_002.features.npy
```

### Spellbook schema sketch

```json
{
  "version": 1,
  "spells": [
    {
      "id": "uuid",
      "name": "Flipendo",
      "enabled": true,
      "modalities": {
        "gesture": true,
        "voice": true
      },
      "osc": {
        "mode": "bool_pulse",
        "address": "/avatar/parameters/osc-grimoire-spell-1",
        "pulse_ms": 250
      },
      "recognition": {
        "gesture_threshold": 0.35,
        "gesture_margin": 0.08,
        "voice_threshold": 42.0,
        "voice_margin": 5.0,
        "requires_both_modalities": true
      },
      "samples": {
        "gestures": [
          "samples/spell_uuid/gesture_001.json"
        ],
        "voices": [
          "samples/spell_uuid/voice_001.wav"
        ]
      }
    }
  ]
}
```

### Config schema sketch

```json
{
  "openvr": {
    "overlay_hand": "left",
    "casting_hand": "right",
    "overlay_width_m": 0.25
  },
  "audio": {
    "input_device": null,
    "sample_rate": 16000,
    "channels": 1
  },
  "osc": {
    "host": "127.0.0.1",
    "port": 9000,
    "emit_casting": true,
    "casting_address": "/avatar/parameters/osc-grimoire-casting"
  },
  "wand": {
    "controller_to_tip_offset": [0.0, 0.0, 0.0]
  }
}
```

## Python implementation notes

Likely useful libraries:

* OpenVR:

  * `openvr` / `pyopenvr`
* OSC:

  * `python-osc`
* Audio input:

  * `sounddevice`
  * `numpy`
  * `scipy`
* Audio features:

  * `librosa`
  * or smaller MFCC-specific libraries if avoiding librosa dependency weight
* DTW:

  * `fastdtw`
  * `dtaidistance`
  * or custom constrained DTW for short clips
* UI/rendering:

  * simplest path: desktop/web UI rendered to texture if feasible
  * alternative: OpenVR overlay with a lightweight immediate-mode style UI
  * for MVP, a desktop companion UI plus minimal VR overlay is acceptable

Do not overinvest in UI before the recognition loop works.

## Suggested architecture

```text
osc_grimoire/
  app.py

  openvr_runtime.py
    - initialize SteamVR/OpenVR
    - read controller poses/buttons
    - manage overlay transform
    - expose trigger/grip events

  overlay_ui.py
    - render spellbook UI
    - handle overlay interactions
    - show recognition/training prompts

  audio_capture.py
    - enumerate devices
    - record while trigger held
    - save clips

  gesture_capture.py
    - collect wand-tip positions while trigger held
    - preprocess traces
    - save gesture samples

  gesture_recognizer.py
    - normalize traces
    - compare against templates
    - return ranked matches

  voice_recognizer.py
    - extract MFCC/log-mel features
    - DTW against templates
    - return ranked matches

  spellbook.py
    - load/save config
    - manage spells/samples
    - threshold settings

  osc_output.py
    - emit VRChat OSC messages
    - pulse bools
    - reset managed params

  calibration.py
    - wand-tip offset calibration
    - controller role management

  debug.py
    - logs
    - score tables
    - sample inspection
```

## Recognition loop pseudocode

```python
def on_cast_start():
    osc.emit_bool(config.osc.casting_address, True)
    gesture_capture.start()
    audio_capture.start_if_needed()

def on_cast_end():
    gesture = gesture_capture.stop()
    audio = audio_capture.stop_if_needed()

    osc.emit_bool(config.osc.casting_address, False)

    query = preprocess_query(gesture, audio)
    ranked = recognizer.match(query, spellbook.enabled_spells)

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    if passes_thresholds(best, second):
        emit_spell(best.spell)
    else:
        ui.show_no_match(ranked)
```

## Training loop pseudocode

```python
def record_training_sample(spell_id, modalities):
    ui.prompt("Hold trigger and perform spell")

    wait_for_trigger_down()

    if modalities.gesture:
        gesture_capture.start()

    if modalities.voice:
        audio_capture.start()

    wait_for_trigger_up()

    if modalities.gesture:
        gesture = gesture_capture.stop()
        spellbook.save_gesture_sample(spell_id, gesture)

    if modalities.voice:
        audio = audio_capture.stop()
        spellbook.save_voice_sample(spell_id, audio)

    spellbook.recompute_spell_features(spell_id)
    spellbook.save()
```

## Threshold calibration

MVP thresholding can be manual, but the app should expose useful diagnostics.

For each test query, show:

```text
best spell: Flipendo
best score: 31.2
second spell: Lumos
second score: 46.8
margin: 15.6
decision: accepted
```

For training samples, compute leave-one-out distances:

```text
For each sample:
  temporarily remove it
  match it against the remaining spellbook
  see whether it returns the correct spell
```

This gives the user a rough “is this spell trainable?” signal.

Useful warnings:

* “This spell is too similar to another spell.”
* “Your samples vary a lot; record more consistent examples.”
* “Threshold too strict; your own training samples fail.”
* “Threshold too loose; this may false-trigger.”

## MVP milestones

### Milestone 1: CLI proof of concept

No VR. No overlay.

* Create spell from command line.
* Record voice samples.
* Record or load gesture traces from simple mouse input or synthetic data.
* Run MFCC+DTW voice matching.
* Run simple gesture template matching.
* Print ranked results.
* Emit OSC manually on match.

Success condition:

```text
User can train 3 fake spoken words and reliably classify held-out recordings.
```

### Milestone 2: VR pose capture

* Read OpenVR controller poses/buttons.
* Capture main-hand trace while trigger is held.
* Save raw trace.
* Visualize trace in desktop debug window.
* Recognize gesture against templates.

Success condition:

```text
User can train 3 gestures in VR and classify new attempts.
```

### Milestone 3: OSC + VRChat integration

* Emit `casting` boolean on trigger hold.
* Emit spell bool pulse on match.
* Confirm VRChat avatar parameter receives values.
* Build a minimal example avatar/controller setup or documentation.

Success condition:

```text
Holding trigger enables avatar wand trail; releasing after a recognized spell triggers avatar effect.
```

### Milestone 4: OpenVR spellbook overlay

* Show offhand quad overlay.
* Display spell list and current mode.
* Support basic training/test prompts.
* Provide visual feedback during capture.

Success condition:

```text
User can train and test spells without taking off headset.
```

### Milestone 5: Calibration and usability

* Add wand-tip offset calibration.
* Add per-spell thresholds.
* Add score diagnostics.
* Add import/export spellbook.
* Add reset-on-start/shutdown OSC safety.

Success condition:

```text
User can configure a small spellbook for a real avatar without editing JSON manually.
```

## Testing plan

### Voice tests

Use deliberately fake words:

* flipendo
* alohomora
* lumos
* nox
* expelliarmus
* made-up nonsense syllables

Test:

* same session vs later session
* quiet room vs fan/noise
* normal voice vs whispered/acted voice
* short vs long trigger window
* near-confusable spells

Metrics:

* top-1 accuracy
* false reject rate
* false accept rate
* margin distribution
* latency after trigger release

### Gesture tests

Use simple glyphs:

* circle
* slash
* triangle
* zigzag
* spiral
* lightning bolt
* horizontal line
* vertical line

Test:

* different drawing speeds
* different sizes
* different body orientations
* raw controller point vs calibrated wand tip
* 2D projection vs 3D matching

Metrics:

* top-1 accuracy
* false reject rate
* confusion matrix
* recognition latency

### VRChat tests

* confirm OSC parameter names match avatar parameters
* confirm bool pulse duration works with animator transitions
* confirm casting boolean resets correctly
* confirm app crash does not leave avatar stuck if restarted
* test with SteamVR overlay active over VRChat

## Design risks

### False positives

Button-gating greatly reduces this, but recognition can still choose a wrong spell.

Mitigations:

* best-vs-second-best margin
* modality agreement for gesture + voice spells
* stricter thresholds for powerful effects
* optional “confirm spell” debug mode
* cooldown after a cast

### Inconsistent user performance

Users may say or draw a spell differently each time.

Mitigations:

* show training consistency diagnostics
* encourage 5–10 samples
* allow multiple templates per spell
* allow per-spell thresholds
* support retraining bad samples

### Similar spells

Some incantations or glyphs may be too close acoustically/geometrically.

Mitigations:

* warning when two spells have overlapping score distributions
* require gesture + voice for similar spells
* encourage more distinct spell names/glyphs

### OpenVR overlay complexity

OpenVR overlays can be fiddly, especially input and transforms.

Mitigations:

* start with desktop/CLI recognizer
* keep overlay UI minimal
* allow keyboard/desktop fallback
* separate overlay code from recognition core

### Python packaging

SteamVR, audio devices, and native libraries can make packaging awkward.

Mitigations:

* keep dependencies boring
* document dev setup clearly
* use a venv/uv/poetry workflow
* later package with PyInstaller or similar only after MVP stabilizes

## Future directions

### Neural embedding recognizer

Replace MFCC+DTW with a pretrained audio embedding model.

Training mode:

```text
record examples
→ embed each clip
→ average embeddings into spell prototype
→ nearest-neighbor/prototype classification
```

Runtime:

```text
query clip
→ embedding
→ nearest spell prototype
→ threshold + margin
```

Potential benefits:

* better robustness to speed/tone variation
* better noise tolerance
* faster matching with many spells

Keep the same UX and data model so this can be swapped in later.

### Vector database

Probably unnecessary for MVP. A spellbook will usually have fewer than 100 spells and fewer than 1000 samples. Brute-force nearest-neighbor is fine.

A vector index only becomes useful if:

* sharing large spell libraries
* comparing many user samples
* storing many embeddings over time
* doing analytics/debugging across users

### Shared spell packs

Allow users to export/import spell definitions:

* OSC bindings
* gesture templates
* optional voice templates

Voice templates are speaker-dependent, so imported voice samples should be treated as examples/placeholders, not reliable detectors.

### Avatar authoring helpers

Generate a VRChat avatar parameter list or documentation snippet from the spellbook.

Example:

```text
osc-grimoire-casting: bool
osc-grimoire-spell-1: bool
osc-grimoire-spell-2: bool
osc-grimoire-spell-index: int
```

Could also generate example Animator Controller notes.

### Local-only visual effects

Because the app is an OpenVR overlay, it may be able to render local-only helper effects:

* casting guide lines
* recognized glyph preview
* debug trail
* spellbook particles

VRChat-visible effects still need to be avatar/world driven through OSC parameters.

## Initial implementation bias

Start boring:

* Python
* OpenVR for poses and overlay
* `python-osc` for OSC
* `sounddevice` for microphone capture
* MFCC + DTW for voice
* `$Q`/point-cloud-style recognizer for projected gestures
* JSON spellbook
* desktop debug UI before polished VR UI

The main unknown is not whether the architecture is possible. It is whether the recognition quality is good enough with a tiny number of user examples. The fastest way to answer that is a CLI/desktop recognizer spike before investing heavily in the VR overlay.
