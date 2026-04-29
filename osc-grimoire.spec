from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


datas = []
binaries = []
hiddenimports = []

for package in (
    "glfw",
    "huggingface_hub",
    "imgui_bundle",
    "onnx_asr",
    "openvr",
    "OpenGL",
    "pythonosc",
    "pythonoscquery",
    "sounddevice",
    "soundfile",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("osc_grimoire")

model_dir = Path("vendor/models/parakeet-ctc-110m-int8")
if model_dir.exists():
    datas.append((str(model_dir), "models/parakeet-ctc-110m-int8"))
datas.append(("THIRD_PARTY_NOTICES.txt", "."))


a = Analysis(
    ["scripts/osc_grimoire_overlay_entry.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="osc-grimoire-overlay",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="osc-grimoire",
)
