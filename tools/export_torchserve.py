import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model artifacts for TorchServe.")
    parser.add_argument(
        "--model_dir",
        default="outputs/dvc_run/model",
        help="Path to the Hugging Face model directory.",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="torchserve/artifacts",
        help="Output directory for TorchServe artifacts.",
    )
    parser.add_argument(
        "--model_store",
        default="torchserve/model-store",
        help="Directory for TorchServe .mar archives.",
    )
    parser.add_argument(
        "--handler",
        default="torchserve/handler.py",
        help="TorchServe handler path.",
    )
    parser.add_argument(
        "--model_name",
        default="mymodel",
        help="Model name for the TorchServe archive.",
    )
    parser.add_argument(
        "--version",
        default="1.0",
        help="Version string for the TorchServe archive.",
    )
    return parser.parse_args()


def _copy_tokenizer_files(model_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for filename in TOKENIZER_FILES:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)
            copied.append(filename)
    return copied


def _run_archiver(
    *,
    model_name: str,
    version: str,
    serialized_file: Path,
    handler: Path,
    export_path: Path,
    extra_files: list[str],
) -> None:
    cmd = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        version,
        "--serialized-file",
        str(serialized_file),
        "--handler",
        str(handler),
        "--export-path",
        str(export_path),
        "--force",
    ]
    if extra_files:
        cmd.extend(["--extra-files", ",".join(extra_files)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    artifacts_dir = Path(args.artifacts_dir)
    model_store = Path(args.model_store)
    handler = Path(args.handler)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_store.mkdir(parents=True, exist_ok=True)

    AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model_path = artifacts_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    copied_files = _copy_tokenizer_files(model_dir, artifacts_dir)
    extra_files = [
        str(artifacts_dir / filename)
        for filename in copied_files
        if (artifacts_dir / filename).exists()
    ]

    _run_archiver(
        model_name=args.model_name,
        version=args.version,
        serialized_file=model_path,
        handler=handler,
        export_path=model_store,
        extra_files=extra_files,
    )

    print("Copied files:", ", ".join(sorted(copied_files)) or "(none)")
    print("Artifacts directory:", artifacts_dir)
    print("Model archive:", model_store / f"{args.model_name}.mar")


if __name__ == "__main__":
    main()
