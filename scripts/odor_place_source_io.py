"""Bounded source I/O for the pinned odor-place feasibility audit."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def check_space(root: Path, remaining_bytes: int, reserve_bytes: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < remaining_bytes + reserve_bytes:
        raise ValueError(f"Insufficient space: {free} available; {remaining_bytes} download + {reserve_bytes} reserve required")
    return free


def download_verified(url: str, target: Path, size: int, expected: str, algorithm: str, reserve: int) -> dict:
    if target.exists():
        if target.stat().st_size != size or digest(target, algorithm) != expected:
            raise ValueError(f"Existing file differs from published digest: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        completed = partial.stat().st_size if partial.exists() else 0
        if completed > size:
            raise ValueError("Partial download is larger than published file")
        check_space(target.parent, size - completed, reserve)
        subprocess.run(["curl", "--fail", "--location", "--retry", "8", "--retry-delay", "5", "--connect-timeout", "30",
                        "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-", "--output", str(partial), url], check=True)
        if partial.stat().st_size != size or digest(partial, algorithm) != expected:
            raise ValueError(f"Published {algorithm} verification failed; partial preserved: {partial}")
        partial.replace(target)
    return {"path": str(target.resolve()), "size_bytes": size, "published_algorithm": algorithm,
            "published_digest": expected, "published_digest_verified": True, "sha256": digest(target)}


class BoundedHTTPFile(io.RawIOBase):
    """Seekable HTTP ranges with a hard transfer budget, never a full-file fallback."""

    def __init__(self, url: str, size: int, budget: int, *, block_size: int = 1024 * 1024):
        self.url, self.size, self.budget, self.block_size = url, size, budget, block_size
        self.position = 0
        self.transferred_bytes = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("Invalid range seek")
        self.position = position
        return position

    def readinto(self, destination):
        content = self.read(len(destination))
        destination[:len(content)] = content
        return len(content)

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        end = min(self.size, self.position + length)
        pieces = []
        while self.position < end:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                stop = min(start + self.block_size, self.size) - 1
                count = stop - start + 1
                if self.transferred_bytes + count > self.budget:
                    raise ValueError("Bounded header transfer budget exceeded")
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{stop}":
                        raise ValueError("Server did not honor requested bounded byte range")
                    data = response.read(count + 1)
                if len(data) != count:
                    raise ValueError("Incorrect range response size")
                self.transferred_bytes += count
                self.cache[index] = data
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            data = self.cache[index]
            offset = self.position % self.block_size
            take = min(len(data) - offset, end - self.position)
            pieces.append(data[offset:offset + take])
            self.position += take
        return b"".join(pieces)
