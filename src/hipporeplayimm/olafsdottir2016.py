"""Utilities for the Olafsdottir/Carpenter/Barry 2016 Axona dataset."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import hashlib
from pathlib import Path
import re
import shutil
from urllib.request import urlopen
import zipfile

import numpy as np


ZENODO_URL = "https://zenodo.org/records/5566548/files/Olafsdottir2016.zip"
EXPECTED_MD5 = "93a9288449bd28d74ce4ff7ed7c6f878"
ARCHIVE_NAME = "Olafsdottir2016.zip"
MANIFEST_NAME = "olafsdottir2016_manifest.csv"
MANIFEST_COLUMNS = [
    "animal",
    "date",
    "session_type",
    "session_name",
    "session_path",
    "has_pos",
    "has_set",
    "n_cut_files",
    "n_egf_files",
    "n_tetrode_files",
    "hippocampal_tetrodes",
    "mec_tetrodes",
    "notes",
]


@dataclass(frozen=True)
class TetrodeArrangement:
    hippocampal_tetrodes: tuple[int, ...]
    mec_tetrodes: tuple[int, ...]
    notes: str


@dataclass(frozen=True)
class OlafsdottirSessionRecord:
    animal: str
    date: str
    session_type: str
    session_name: str
    session_path: str
    has_pos: bool
    has_set: bool
    n_cut_files: int
    n_egf_files: int
    n_tetrode_files: int
    hippocampal_tetrodes: tuple[int, ...]
    mec_tetrodes: tuple[int, ...]
    notes: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "animal": self.animal,
            "date": self.date,
            "session_type": self.session_type,
            "session_name": self.session_name,
            "session_path": self.session_path,
            "has_pos": str(bool(self.has_pos)).lower(),
            "has_set": str(bool(self.has_set)).lower(),
            "n_cut_files": str(int(self.n_cut_files)),
            "n_egf_files": str(int(self.n_egf_files)),
            "n_tetrode_files": str(int(self.n_tetrode_files)),
            "hippocampal_tetrodes": _format_tetrodes(self.hippocampal_tetrodes),
            "mec_tetrodes": _format_tetrodes(self.mec_tetrodes),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AxonaPosition:
    header: dict[str, str]
    times_s: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    x_cm: np.ndarray
    y_cm: np.ndarray
    valid: np.ndarray
    pixels_per_metre: float


@dataclass(frozen=True)
class AxonaCut:
    labels: np.ndarray
    spike_times_s: np.ndarray | None = None


@dataclass(frozen=True)
class AxonaEgf:
    header: dict[str, str]
    signal: np.ndarray
    sample_rate_hz: float
    times_s: np.ndarray


def read_axona_set(path: str | Path) -> dict[str, str]:
    """Read an Axona .set header as a string dictionary."""

    header, _payload = _read_axona_header_and_payload(path)
    return header


def read_axona_pos(path: str | Path) -> AxonaPosition:
    """Read the subset of Axona .pos needed for 1D/Z-track encoding."""

    header, payload = _read_axona_header_and_payload(path)
    timestamp_bytes = _header_int(header, "bytes_per_timestamp", 4)
    coord_bytes = _header_int(header, "bytes_per_coord", 2)
    if timestamp_bytes != 4 or coord_bytes != 2:
        raise ValueError("Only Axona .pos files with 4-byte timestamps and 2-byte coordinates are supported")
    record_size = timestamp_bytes + 8 * coord_bytes
    n_records = _payload_record_count(payload, record_size, _header_int(header, "num_pos_samples", 0))
    dtype = np.dtype(
        [
            ("t", ">u4"),
            ("x1", ">i2"),
            ("y1", ">i2"),
            ("x2", ">i2"),
            ("y2", ">i2"),
            ("numpix1", ">i2"),
            ("numpix2", ">i2"),
            ("totalpix1", ">i2"),
            ("totalpix2", ">i2"),
        ]
    )
    records = np.frombuffer(payload[: n_records * record_size], dtype=dtype)
    timebase_hz = _header_float(header, "timebase", _header_float(header, "sample_rate", 50.0))
    times_s = records["t"].astype(float) / float(timebase_hz)
    if times_s.size:
        times_s = times_s - float(times_s[0])
    pixels_per_metre = _header_float(header, "pixels_per_metre", 100.0)
    x1 = records["x1"].astype(float)
    y1 = records["y1"].astype(float)
    x2 = records["x2"].astype(float)
    y2 = records["y2"].astype(float)
    valid1 = _valid_led_xy(x1, y1)
    valid2 = _valid_led_xy(x2, y2)
    x = np.full(n_records, np.nan, dtype=float)
    y = np.full(n_records, np.nan, dtype=float)
    both = valid1 & valid2
    x[both] = 0.5 * (x1[both] + x2[both])
    y[both] = 0.5 * (y1[both] + y2[both])
    only1 = valid1 & ~valid2
    only2 = valid2 & ~valid1
    x[only1] = x1[only1]
    y[only1] = y1[only1]
    x[only2] = x2[only2]
    y[only2] = y2[only2]
    scale = 100.0 / float(pixels_per_metre)
    return AxonaPosition(
        header=header,
        times_s=times_s,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        x_cm=x * scale,
        y_cm=y * scale,
        valid=np.isfinite(x) & np.isfinite(y),
        pixels_per_metre=float(pixels_per_metre),
    )


def read_axona_cut(path: str | Path, tetrode_path: str | Path | None = None) -> AxonaCut:
    """Read Axona .cut labels, optionally attaching spike times from a tetrode file."""

    text = Path(path).read_text(encoding="latin-1", errors="ignore")
    lines = text.splitlines()
    start_index = None
    expected_spikes = None
    for index, line in enumerate(lines):
        if "Exact_cut_for" in line and "spikes:" in line:
            start_index = index + 1
            match = re.search(r"spikes:\s*(\d+)", line)
            expected_spikes = int(match.group(1)) if match else None
            break
    numeric_lines = lines[start_index:] if start_index is not None else lines
    labels = np.asarray(
        [int(value) for line in numeric_lines for value in re.findall(r"[-+]?\d+", line)],
        dtype=int,
    )
    if expected_spikes is not None:
        labels = labels[:expected_spikes]
    spike_times_s = read_axona_tetrode_spike_times(tetrode_path) if tetrode_path is not None else None
    if spike_times_s is not None and spike_times_s.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Cut labels ({labels.shape[0]}) do not match tetrode spike times ({spike_times_s.shape[0]})"
        )
    return AxonaCut(labels=labels, spike_times_s=spike_times_s)


def read_axona_egf(path: str | Path) -> AxonaEgf:
    """Read Axona .egf int16 LFP samples."""

    header, payload = _read_axona_header_and_payload(path)
    n_samples = len(payload) // 2
    signal = np.frombuffer(payload[: n_samples * 2], dtype=">i2").astype(np.int16)
    sample_rate_hz = _header_float(header, "sample_rate", 4800.0)
    times_s = np.arange(signal.shape[0], dtype=float) / float(sample_rate_hz)
    return AxonaEgf(
        header=header,
        signal=signal,
        sample_rate_hz=float(sample_rate_hz),
        times_s=times_s,
    )


def read_axona_tetrode_spike_times(path: str | Path) -> np.ndarray:
    """Read spike timestamps from a raw Axona tetrode file."""

    header, payload = _read_axona_header_and_payload(path)
    samples_per_spike = _header_int(header, "samples_per_spike", 50)
    record_size = 4 + 4 * samples_per_spike
    n_spikes = _payload_record_count(payload, record_size, _header_int(header, "num_spikes", 0))
    if n_spikes == 0:
        return np.empty(0, dtype=float)
    records = np.frombuffer(payload[: n_spikes * record_size], dtype=np.uint8).reshape(n_spikes, record_size)
    timestamps = np.ascontiguousarray(records[:, :4]).view(">u4").reshape(-1).astype(float)
    timebase_hz = _header_float(header, "timebase", 96000.0)
    return timestamps / float(timebase_hz)



def tetrode_arrangement_for_animal(animal: str) -> TetrodeArrangement:
    """Return MEC/HPC tetrode assignment from the Zenodo dataset note."""

    normalized = animal.strip().upper()
    if normalized == "R2142":
        return TetrodeArrangement(
            hippocampal_tetrodes=tuple(range(1, 9)),
            mec_tetrodes=tuple(range(9, 17)),
            notes="R2142 reversed arrangement: hippocampus=1-8, MEC=9-16",
        )
    return TetrodeArrangement(
        hippocampal_tetrodes=tuple(range(9, 17)),
        mec_tetrodes=tuple(range(1, 9)),
        notes="standard arrangement: MEC=1-8, hippocampus=9-16",
    )


def build_manifest(dataset_root: str | Path) -> list[OlafsdottirSessionRecord]:
    """Scan extracted Axona files and return one manifest record per session stem."""

    root = Path(dataset_root)
    if not root.exists():
        return []
    records: list[OlafsdottirSessionRecord] = []
    for animal_dir in sorted(path for path in root.iterdir() if path.is_dir() and _looks_like_animal_dir(path)):
        animal = animal_dir.name.upper()
        arrangement = tetrode_arrangement_for_animal(animal)
        for date_dir in sorted(path for path in animal_dir.iterdir() if path.is_dir()):
            stems = _session_stems(date_dir)
            for stem in sorted(stems):
                session_type = infer_session_type(stem)
                records.append(
                    OlafsdottirSessionRecord(
                        animal=animal,
                        date=date_dir.name,
                        session_type=session_type,
                        session_name=stem,
                        session_path=str(date_dir / stem),
                        has_pos=(date_dir / f"{stem}.pos").exists(),
                        has_set=(date_dir / f"{stem}.set").exists(),
                        n_cut_files=_count_cut_files(date_dir, stem),
                        n_egf_files=_count_egf_files(date_dir, stem),
                        n_tetrode_files=_count_tetrode_files(date_dir, stem),
                        hippocampal_tetrodes=arrangement.hippocampal_tetrodes,
                        mec_tetrodes=arrangement.mec_tetrodes,
                        notes=arrangement.notes,
                    )
                )
    return records


def write_manifest_csv(records: list[OlafsdottirSessionRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())
    return path


def prepare_dataset(
    *,
    dataset_root: str | Path,
    zenodo_url: str = ZENODO_URL,
    expected_md5: str = EXPECTED_MD5,
    archive_path: str | Path | None = None,
    manifest_output: str | Path | None = None,
    download: bool = False,
    extract: bool = False,
    force_download: bool = False,
) -> tuple[Path, list[OlafsdottirSessionRecord]]:
    """Optionally download/extract the archive, then write the session manifest."""

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(archive_path) if archive_path is not None else root / ARCHIVE_NAME
    if download and (force_download or not archive.exists()):
        download_archive(zenodo_url, archive)
    if archive.exists() and expected_md5:
        verify_md5(archive, expected_md5)
    if extract:
        if not archive.exists():
            raise FileNotFoundError(
                f"Cannot extract {archive}; pass --download or provide --archive-path."
            )
        extract_archive(archive, root)
    records = build_manifest(root)
    output = Path(manifest_output) if manifest_output is not None else root / MANIFEST_NAME
    return write_manifest_csv(records, output), records


def download_archive(url: str, output_path: str | Path) -> Path:
    """Stream the Zenodo archive to disk without loading it into memory."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp_path.replace(path)
    return path


def md5_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: str | Path, expected_md5: str) -> None:
    observed = md5_file(path)
    expected = expected_md5.lower()
    if observed.lower() != expected:
        raise ValueError(f"MD5 mismatch for {path}: observed {observed}, expected {expected}")


def extract_archive(archive_path: str | Path, dataset_root: str | Path) -> None:
    root = Path(dataset_root)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)


def _read_axona_header_and_payload(path: str | Path) -> tuple[dict[str, str], bytes]:
    payload = Path(path).read_bytes()
    marker = b"data_start"
    index = payload.find(marker)
    if index < 0:
        return _parse_axona_header(payload.decode("latin-1", errors="ignore")), b""
    data_offset = index + len(marker)
    if payload[data_offset : data_offset + 2] == b"\r\n":
        data_offset += 2
    elif payload[data_offset : data_offset + 1] in {b"\r", b"\n"}:
        data_offset += 1
    header = _parse_axona_header(payload[:data_offset].decode("latin-1", errors="ignore"))
    return header, _strip_axona_data_end(payload[data_offset:])


def _strip_axona_data_end(payload: bytes) -> bytes:
    """Strip an optional Axona ``data_end`` footer without touching binary samples."""

    marker = b"data_end"
    end = len(payload)
    while end > 0 and payload[end - 1 : end] in {b"\r", b"\n", b"\t", b" "}:
        end -= 1
    if not payload[:end].endswith(marker):
        return payload
    footer_start = end - len(marker)
    if payload[max(0, footer_start - 2) : footer_start] == b"\r\n":
        footer_start -= 2
    elif payload[max(0, footer_start - 1) : footer_start] in {b"\r", b"\n"}:
        footer_start -= 1
    return payload[:footer_start]


def _parse_axona_header(text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\x00")
        if not line or line == "data_start":
            continue
        if " " in line:
            key, value = line.split(None, 1)
            header[key] = value.strip().strip("\x00")
        else:
            header[line] = ""
    return header


def _header_float(header: dict[str, str], key: str, default: float) -> float:
    raw = header.get(key)
    if raw is None:
        return float(default)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    return float(match.group(0)) if match else float(default)


def _header_int(header: dict[str, str], key: str, default: int) -> int:
    value = _header_float(header, key, float(default))
    if not math.isfinite(value):
        return int(default)
    return int(round(value))


def _payload_record_count(payload: bytes, record_size: int, header_count: int) -> int:
    available = len(payload) // int(record_size)
    if header_count <= 0:
        return available
    return min(int(header_count), available)


def _valid_led_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= 0.0) & (x < 1023.0) & (y >= 0.0) & (y < 1023.0)


def infer_session_type(session_name: str) -> str:
    lower = session_name.lower()
    if "sleeppost" in lower:
        return "sleepPOST"
    if "track1" in lower:
        return "track1"
    if "training" in lower:
        return "Training"
    if "screening" in lower:
        return "Screening"
    tokens = lower.split("_")
    return tokens[-1] if tokens else lower


def _session_stems(date_dir: Path) -> set[str]:
    stems: set[str] = set()
    for path in date_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if ".clu." in name.lower() or name.lower().endswith(".clu"):
            continue
        if name.endswith(".set") or name.endswith(".pos"):
            stems.add(path.stem)
            continue
        egf = re.match(r"^(?P<stem>.+)\.egf\d*$", name, flags=re.IGNORECASE)
        if egf:
            stems.add(egf.group("stem"))
            continue
        tetrode = re.match(r"^(?P<stem>.+)\.\d+$", name)
        if tetrode:
            stems.add(tetrode.group("stem"))
            continue
        cut = re.match(r"^(?P<stem>.+?)(?:_\d+)?\.cut$", name, flags=re.IGNORECASE)
        if cut:
            stems.add(cut.group("stem"))
    return stems


def _count_cut_files(date_dir: Path, stem: str) -> int:
    return sum(
        1
        for path in date_dir.iterdir()
        if path.is_file()
        and re.match(rf"^{re.escape(stem)}(?:_\d+)?\.cut$", path.name, flags=re.IGNORECASE)
    )


def _count_egf_files(date_dir: Path, stem: str) -> int:
    return sum(
        1
        for path in date_dir.iterdir()
        if path.is_file()
        and re.match(rf"^{re.escape(stem)}\.egf\d*$", path.name, flags=re.IGNORECASE)
    )


def _count_tetrode_files(date_dir: Path, stem: str) -> int:
    return sum(
        1
        for path in date_dir.iterdir()
        if path.is_file() and re.match(rf"^{re.escape(stem)}\.\d+$", path.name)
    )


def _looks_like_animal_dir(path: Path) -> bool:
    return bool(re.match(r"^r\d+$", path.name, flags=re.IGNORECASE))


def _format_tetrodes(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)
