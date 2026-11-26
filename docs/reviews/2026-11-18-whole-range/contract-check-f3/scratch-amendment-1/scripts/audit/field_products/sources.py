"""The source freeze (code review r1 F1). Every input is read exactly once into memory and hashed before any outcome
is parsed; parsers get bytes only through ``Stage.read`` (an unstaged path is refused, so nothing outcome-bearing can
be read after the freeze); ``verify`` re-reads the disk at the end and reports any content or listing change, which the
driver turns into a refusal before writing results."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Stage:
    def __init__(self) -> None:
        self._data: dict[str, bytes | None] = {}
        self._keys: dict[str, str] = {}
        self.files: dict[str, dict] = {}
        self.listings: dict[str, dict] = {}

    def add(self, path: Path, key: str) -> bytes | None:
        p = str(Path(path).resolve())
        try:
            b = Path(p).read_bytes()
        except FileNotFoundError:
            b = None
        if p in self._data:
            if self._data[p] != b:
                raise ValueError(f"{p} already staged with different bytes")
            return b
        self._data[p], self._keys[p] = b, key
        self.files[key] = {"path": p, "sha256": _sha(b) if b is not None else None,
                           "bytes": len(b) if b is not None else None}
        return b

    def read(self, path: Path) -> bytes:
        p = str(Path(path).resolve())
        if p not in self._data:
            raise KeyError(f"not staged: {p}")
        if self._data[p] is None:
            raise FileNotFoundError(p)
        return self._data[p]

    def sha(self, path: Path) -> str | None:
        return self.files[self._keys[str(Path(path).resolve())]]["sha256"]

    @staticmethod
    def _names(directory: Path, pattern: str, recursive: bool) -> list[str]:
        d = Path(directory)
        it = d.rglob(pattern) if recursive else d.glob(pattern)
        return sorted(str(p.relative_to(d)) for p in it if p.is_file())

    def listing(self, key: str, directory: Path, pattern: str = "*", recursive: bool = False) -> list[str]:
        names = self._names(directory, pattern, recursive)
        self.listings[key] = {"dir": str(Path(directory).resolve()), "pattern": pattern, "recursive": recursive,
                              "n": len(names), "names_sha256": _sha("\n".join(names).encode())}
        return names

    def manifest(self) -> dict:
        return {"files": dict(sorted(self.files.items())), "listings": dict(sorted(self.listings.items()))}

    def verify(self) -> list[str]:
        problems = []
        for key, f in self.files.items():
            try:
                now = _sha(Path(f["path"]).read_bytes())
            except FileNotFoundError:
                now = None
            if now != f["sha256"]:
                problems.append(f"{key}: content changed ({f['sha256']} -> {now})")
        for key, ls in self.listings.items():
            names = self._names(Path(ls["dir"]), ls["pattern"], ls["recursive"])
            if _sha("\n".join(names).encode()) != ls["names_sha256"]:
                problems.append(f"{key}: listing changed")
        return problems
