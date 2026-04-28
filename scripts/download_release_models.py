from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Systran/faster-whisper-tiny"
TARGET_DIR = Path("vendor/models/faster-whisper-tiny")


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        allow_patterns=[
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.txt",
            "preprocessor_config.json",
        ],
    )
    print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
