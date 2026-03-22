#!/usr/bin/env python3

import json
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "build" / "compile_commands.json"
    dst = repo / ".vscode" / "compile_commands.json"

    if not src.is_file():
        raise SystemExit(f"missing compile commands: {src}")

    entries = json.loads(src.read_text())
    variants = []
    seen = set()

    path_pairs = [
        ("/home/sfengzhe/scratch/ic", "/scratch/sfengzhe/ic"),
        ("/scratch/sfengzhe/ic", "/home/sfengzhe/scratch/ic"),
    ]

    for entry in entries:
        for old, new in [("", "")] + path_pairs:
            item = dict(entry)
            for key in ("directory", "file", "command", "output"):
                if key in item and isinstance(item[key], str) and old:
                    item[key] = item[key].replace(old, new)
            marker = tuple(item.get(k, "") for k in ("directory", "file", "command", "output"))
            if marker not in seen:
                seen.add(marker)
                variants.append(item)

    dst.write_text(json.dumps(variants, indent=2) + "\n")


if __name__ == "__main__":
    main()
