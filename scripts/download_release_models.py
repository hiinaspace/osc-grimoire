from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "entropora/parakeet-ctc-110m-int8"
TARGET_DIR = Path("vendor/models/parakeet-ctc-110m-int8")


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(hf_hub_download(MODEL_ID, "encoder.int8.onnx")).resolve()
    tokens_path = Path(hf_hub_download(MODEL_ID, "tokens.txt")).resolve()
    (TARGET_DIR / "model.onnx").write_bytes(model_path.read_bytes())
    (TARGET_DIR / "vocab.txt").write_bytes(tokens_path.read_bytes())
    (TARGET_DIR / "config.json").write_text(
        (
            '{"model_type":"nemo-conformer-ctc",'
            '"subsampling_factor":8,'
            '"features_size":80}'
        ),
        encoding="utf-8",
    )
    print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
