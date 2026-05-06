#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "matrices"


@dataclass(frozen=True)
class MatrixSource:
    group: str
    name: str
    output_subdir: str
    output_file: str

    @property
    def urls(self) -> list[str]:
        return [
            f"https://suitesparse-collection-website.herokuapp.com/MM/{self.group}/{self.name}.tar.gz",
            f"https://sparse.tamu.edu/MM/{self.group}/{self.name}.tar.gz",
            f"http://sparse-files.engr.tamu.edu/MM/{self.group}/{self.name}.tar.gz",
        ]


MATRICES: list[MatrixSource] = [
    MatrixSource("HB", "bcsstk21", "bcsstk21", "bcsstk21.mtx"),
    MatrixSource("UTEP", "Dubcova1", "Dubcova1", "Dubcova1.mtx"),
    MatrixSource("UTEP", "Dubcova2", "Dubcova2", "Dubcova2.mtx"),
    MatrixSource("Norris", "fv1", "fv1", "fv1.mtx"),
    MatrixSource("MathWorks", "Kuu", "kuu", "kuu.mtx"),
    MatrixSource("Nasa", "nasa2146", "nasa2146", "nasa2146.mtx"),
    MatrixSource("ACUSIM", "Pres_Poisson", "Pres_Poisson", "Pres_Poisson.mtx"),
]


def download_file(matrix: MatrixSource, archive_path: Path) -> None:
    last_error: Exception | None = None
    for url in matrix.urls:
        try:
            print(f"[download] {matrix.group}/{matrix.name}: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                with archive_path.open("wb") as file:
                    shutil.copyfileobj(response, file)
            if archive_path.stat().st_size == 0:
                raise RuntimeError("downloaded empty archive")
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            print(f"[warn] failed: {exc}")
    raise RuntimeError(f"failed to download {matrix.group}/{matrix.name}") from last_error


def find_mtx_file(extract_dir: Path, expected_name: str) -> Path:
    matches = sorted(extract_dir.rglob(expected_name))
    if matches:
        return matches[0]
    all_mtx = sorted(extract_dir.rglob("*.mtx"))
    if len(all_mtx) == 1:
        return all_mtx[0]
    raise FileNotFoundError(f"cannot find {expected_name} in extracted archive")


def install_matrix(matrix: MatrixSource, output_dir: Path, force: bool) -> Path:
    destination_dir = output_dir / matrix.output_subdir
    destination_path = destination_dir / matrix.output_file
    if destination_path.exists() and not force:
        print(f"[skip] {destination_path} already exists")
        return destination_path

    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{matrix.name}_") as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / f"{matrix.name}.tar.gz"
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        download_file(matrix, archive_path)
        print(f"[extract] {archive_path.name}")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(extract_dir)
        source_path = find_mtx_file(extract_dir, matrix.output_file)
        shutil.copy2(source_path, destination_path)
    print(f"[ok] {destination_path}")
    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download demo matrices from SuiteSparse Matrix Collection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="matrix output directory, default: data/matrices",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing matrix files")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for matrix in MATRICES:
        install_matrix(matrix, output_dir, args.force)

    print("\nAll matrices are ready:")
    for matrix in MATRICES:
        print(output_dir / matrix.output_subdir / matrix.output_file)


if __name__ == "__main__":
    main()
