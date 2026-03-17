from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotrandhd.config import load_settings
from autotrandhd.services import ModelRegistry, OCRPipelineService
from autotrandhd.utils.image_io import collect_image_paths
from autotrandhd.utils.runtime import build_runtime_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotrandhd-cli", description="OCR tooling for historical Spanish print")
    subparsers = parser.add_subparsers(dest="command")

    infer_parser = subparsers.add_parser("infer", help="Transcribe a single image")
    infer_parser.add_argument("image_path")
    infer_parser.add_argument("--beam-width", type=int, default=10)

    batch_parser = subparsers.add_parser("batch", help="Transcribe a directory of images")
    batch_parser.add_argument("folder_path")
    batch_parser.add_argument("--beam-width", type=int, default=10)

    subparsers.add_parser("info", help="Print runtime metadata")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()
    registry = ModelRegistry(settings)
    service = OCRPipelineService(settings, registry)

    if args.command == "info":
        print(json.dumps(build_runtime_snapshot(), indent=2))
        return

    if args.command == "infer":
        result = service.transcribe_path(Path(args.image_path), beam_width=args.beam_width)
        print(json.dumps(result, indent=2))
        return

    if args.command == "batch":
        results = []
        for image_path in collect_image_paths(args.folder_path):
            try:
                results.append(service.transcribe_path(image_path, beam_width=args.beam_width))
            except Exception as exc:
                results.append({"image": image_path.name, "error": str(exc)})
        print(json.dumps(results, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
