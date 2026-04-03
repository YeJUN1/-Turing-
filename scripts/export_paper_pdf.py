#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CHROME_CANDIDATES = [
    "google-chrome",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_executable(name: str, candidates: list[str] | None = None) -> str | None:
    if candidates is None:
        return shutil.which(name)
    for candidate in candidates:
        if candidate.startswith("/"):
            if Path(candidate).exists():
                return candidate
        else:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Markdown paper to PDF using pandoc and headless Chrome."
    )
    parser.add_argument(
        "--input",
        default="docs/paper_design.md",
        help="Input Markdown file. Default: docs/paper_design.md",
    )
    parser.add_argument(
        "--output",
        default="docs/paper_design.pdf",
        help="Output PDF file. Default: docs/paper_design.pdf",
    )
    parser.add_argument(
        "--title",
        default="头足类伪装计算模拟论文草稿",
        help="Document title metadata passed to pandoc.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path.cwd()
    input_path = (repo_root / args.input).resolve()
    output_path = (repo_root / args.output).resolve()
    header_path = (repo_root / "docs/assets/paper_pdf_header.html").resolve()

    if not input_path.exists():
        raise SystemExit(f"Input Markdown not found: {input_path}")
    if not header_path.exists():
        raise SystemExit(f"Missing header include file: {header_path}")

    pandoc = find_executable("pandoc")
    if not pandoc:
        raise SystemExit("pandoc is required but was not found in PATH.")

    chrome = find_executable("chrome", CHROME_CANDIDATES)
    if not chrome:
        raise SystemExit(
            "Google Chrome/Chromium is required for PDF export but was not found."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="paper_export_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        html_path = temp_dir_path / f"{input_path.stem}.html"

        pandoc_cmd = [
            pandoc,
            str(input_path.name),
            "--from",
            "markdown+raw_html",
            "--to",
            "html5",
            "--standalone",
            "--embed-resources",
            "--metadata",
            f"title={args.title}",
            "--include-in-header",
            str(header_path),
            "--output",
            str(html_path),
        ]

        run(pandoc_cmd, cwd=input_path.parent)

        chrome_cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={output_path}",
            html_path.as_uri(),
        ]

        try:
            run(chrome_cmd)
        except SystemExit:
            fallback_cmd = [
                chrome,
                "--headless",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={output_path}",
                html_path.as_uri(),
            ]
            run(fallback_cmd)

    print(f"Exported PDF: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
