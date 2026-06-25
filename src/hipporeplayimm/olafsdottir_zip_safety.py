"""Guard Olafsdottir archive extraction against unsafe zip members."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import zipfile

_PATCHED_FLAG = "_olafsdottir_zip_safety_patch_applied"


def _safe_zip_member_path(root: Path, member_name: str) -> Path:
    """Return the resolved output path for a safe zip member name."""

    normalized = str(member_name).replace("\\", "/")
    if "\x00" in normalized:
        raise ValueError(f"Unsafe zip member path: {member_name!r}")
    if not normalized:
        raise ValueError("Unsafe zip member path: empty filename")
    member_path = PurePosixPath(normalized)
    first_part = member_path.parts[0] if member_path.parts else ""
    has_windows_drive = len(first_part) >= 2 and first_part[1] == ":" and first_part[0].isalpha()
    if member_path.is_absolute() or any(part == ".." for part in member_path.parts) or has_windows_drive:
        raise ValueError(f"Unsafe zip member path: {member_name!r}")
    destination = (root / Path(*member_path.parts)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe zip member path: {member_name!r}") from exc
    return destination


def extract_archive(archive_path: str | Path, dataset_root: str | Path) -> None:
    """Extract a zip archive only after validating every member path."""

    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _safe_zip_member_path(root, member.filename)
        archive.extractall(root)


def apply_olafsdottir_zip_safety_patch() -> None:
    """Install the safe archive extractor for Olafsdottir dataset preparation."""

    from . import olafsdottir2016

    if getattr(olafsdottir2016, _PATCHED_FLAG, False):
        return
    olafsdottir2016.extract_archive = extract_archive
    setattr(olafsdottir2016, _PATCHED_FLAG, True)


__all__ = ["apply_olafsdottir_zip_safety_patch", "extract_archive"]
