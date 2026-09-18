from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HF FLEURS + UD information-locality pipeline.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--config", default="additionals/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-languages", type=int, default=None)
    parser.add_argument("--languages", nargs="*", default=None, help="Optional explicit language names from the new 102-language HF registry.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run(config_path=config_path, mode=args.mode, device=args.device, overwrite=args.overwrite,
        languages=args.languages, max_languages=args.max_languages)


if __name__ == "__main__":
    main()
