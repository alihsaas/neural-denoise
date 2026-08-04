import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from scipy.io import loadmat

from neural_denoise.config import (
    ARCHIVE_DIR,
    BLOCKS_DIR,
    DATA_DIR,
    SIDD_SMALL_ARCHIVE,
    SIDD_SMALL_MD5,
    SIDD_SOURCE,
    TEST_CLEAN_FILE,
    TEST_CLEAN_KEY,
    TEST_NOISY_FILE,
    TEST_NOISY_KEY,
)


@dataclass(frozen=True)
class Instance:
    name: str
    scene_instance: int
    scene: int
    phone: str
    iso: int
    shutter: int
    illuminant: int
    brightness: str


@dataclass(frozen=True)
class Pair:
    instance: Instance
    archive: Path
    noisy: str
    clean: str


def parse_instance(name: str) -> Instance:
    parts = name.split("_")
    if len(parts) != 7:
        raise ValueError(f"Not a SIDD scene instance name: {name}")
    return Instance(
        name,
        int(parts[0]),
        int(parts[1]),
        parts[2],
        int(parts[3]),
        int(parts[4]),
        int(parts[5]),
        parts[6],
    )


def find_pairs(archive: Path = ARCHIVE_DIR / SIDD_SMALL_ARCHIVE) -> list[Pair]:
    folders: dict[str, dict[str, list[str]]] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            parts = PurePosixPath(member).parts
            if len(parts) < 3 or parts[-3] != "Data":
                continue
            file_name = parts[-1]
            if not file_name.endswith(".PNG"):
                continue
            if file_name.startswith("NOISY_SRGB_"):
                kind = "noisy"
            elif file_name.startswith("GT_SRGB_"):
                kind = "clean"
            else:
                continue
            folders.setdefault(parts[-2], {"noisy": [], "clean": []})[kind].append(member)
    pairs = []
    for name in sorted(folders):
        found = folders[name]
        if len(found["noisy"]) != 1 or len(found["clean"]) != 1:
            raise ValueError(f"{name}: expected one noisy and one clean PNG")
        pairs.append(Pair(parse_instance(name), archive, found["noisy"][0], found["clean"][0]))
    return pairs


def decode_rgb(data: bytes, label: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {label}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pair(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(pair.archive) as bundle:
        noisy = decode_rgb(bundle.read(pair.noisy), pair.noisy)
        clean = decode_rgb(bundle.read(pair.clean), pair.clean)
    if noisy.shape != clean.shape:
        raise ValueError(f"{pair.instance.name}: noisy {noisy.shape} and clean {clean.shape}")
    return noisy, clean


def digest_of(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_blocks(mat_path: Path, key: str, target: Path) -> tuple[int, ...]:
    blocks = loadmat(str(mat_path))[key]
    if blocks.dtype != np.uint8 or blocks.ndim != 5 or blocks.shape[-1] != 3:
        raise ValueError(f"{mat_path.name}: unexpected array {blocks.dtype} {blocks.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.ascontiguousarray(blocks))
    shape = tuple(int(value) for value in blocks.shape)
    del blocks
    return shape


def refresh_blocks(mat_path: Path, key: str, target: Path) -> str:
    sha256 = digest_of(mat_path, "sha256")
    record_path = target.with_suffix(".json")
    if target.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="ascii"))
        if record.get("source_sha256") == sha256:
            return sha256
    record_path.unlink(missing_ok=True)
    print(f"Converting {mat_path.name}")
    convert_blocks(mat_path, key, target)
    record_path.write_text(
        json.dumps({"source": mat_path.name, "source_sha256": sha256}, indent=2), encoding="ascii"
    )
    return sha256


def load_blocks(name: str, blocks_dir: Path = BLOCKS_DIR) -> np.ndarray:
    return np.load(blocks_dir / f"{name}.npy", mmap_mode="r")


def missing_files(archive_dir: Path = ARCHIVE_DIR) -> list[str]:
    names = (SIDD_SMALL_ARCHIVE, TEST_NOISY_FILE, TEST_CLEAN_FILE)
    return [name for name in names if not (archive_dir / name).exists()]


def ensure_sidd(data_dir: Path = DATA_DIR, archive_dir: Path = ARCHIVE_DIR) -> dict:
    missing = missing_files(archive_dir)
    if missing:
        listed = "\n".join(f"  {name}" for name in missing)
        raise SystemExit(f"Put these SIDD files in {archive_dir} ({SIDD_SOURCE}):\n{listed}")
    record = {}
    archive = archive_dir / SIDD_SMALL_ARCHIVE
    md5 = digest_of(archive, "md5")
    if md5 != SIDD_SMALL_MD5:
        raise SystemExit(f"{archive.name}: MD5 {md5} does not match the published {SIDD_SMALL_MD5}")
    record[archive.name] = {
        "bytes": archive.stat().st_size,
        "md5": md5,
        "sha256": digest_of(archive, "sha256"),
        "pairs": len(find_pairs(archive)),
    }
    for name, key, stem in (
        (TEST_NOISY_FILE, TEST_NOISY_KEY, "test_noisy"),
        (TEST_CLEAN_FILE, TEST_CLEAN_KEY, "test_clean"),
    ):
        path = archive_dir / name
        target = data_dir / BLOCKS_DIR.name / f"{stem}.npy"
        sha256 = refresh_blocks(path, key, target)
        record[name] = {
            "bytes": path.stat().st_size,
            "md5": digest_of(path, "md5"),
            "sha256": sha256,
            "shape": list(np.load(target, mmap_mode="r").shape),
        }
    return record


def write_download_record(record: dict, data_dir: Path = DATA_DIR) -> None:
    (data_dir / "download.json").write_text(json.dumps(record, indent=2), encoding="ascii")
