# OSC Grimoire (WIP)

Do wizard stuff in vrchat with verbal and somatic components.

## How do

Advanced wizards only at the moment:

```
uvx git+https://github.com/hiinaspace/osc-grimoire
```

You'll see a little overlay on your left wrist where you can train up some spells.

## Local release build

Windows release builds are PyInstaller `onedir` bundles with the
`entropora/parakeet-ctc-110m-int8` ONNX model included locally.

```
uv sync --group build
.\scripts\build_release.ps1
```

The build writes `dist\osc-grimoire-windows.zip`. The unpacked executable is
`dist\osc-grimoire\osc-grimoire-overlay.exe`.

## GitHub release build

Pushing a `v*` tag runs the Windows release workflow, uploads the zip artifact,
and creates or updates the matching GitHub release:

```
git tag v0.1.0
git push origin v0.1.0
```

The workflow can also be run manually from GitHub Actions to produce a build
artifact without publishing a release.

### Didst thou consort with demons to make this?

I am the bone of my slop, etc, etc
