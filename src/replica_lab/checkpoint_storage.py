"""Share byte-identical saved native weights through independent APFS clones.

Serialization still happens normally before this optional storage optimization.
The pool is outside self-contained checkpoints; manifests and byte receipts keep
their existing meaning. No tensor identity/version or parameter-freezing claim
is used to decide equality. Pool garbage collection is deliberately separate.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHUNK_BYTES = 8 * 1024 * 1024
NATIVE_WEIGHT = re.compile(r"model(?:-\d+-of-\d+)?\.safetensors\Z")


class StorageIntegrityError(ValueError):
    """An invalid pool entry or clone must never become a saved model."""


def _no_symlinks(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise StorageIntegrityError(f"Checkpoint storage refuses symlink paths: {path}")


def _state(path: Path) -> tuple:
    _no_symlinks(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise StorageIntegrityError(f"Expected an independent regular native weight file: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def file_sha256(path: Path) -> str:
    before = _state(path)
    value = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise StorageIntegrityError(f"Native weight was replaced during hashing: {path}")
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            value.update(chunk)
    if _state(path) != before:
        raise StorageIntegrityError(f"Native weight changed during hashing: {path}")
    return value.hexdigest()


def clone_file(source: Path, destination: Path) -> None:
    """Public clonefile(2), without ordinary-copy or hard-link fallback."""
    if platform.system() != "Darwin":
        raise OSError("APFS clonefile requires macOS")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    call = library.clonefile
    call.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    call.restype = ctypes.c_int
    if call(os.fsencode(source), os.fsencode(destination), 1) != 0:  # CLONE_NOFOLLOW
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


@contextmanager
def _hash_lock(pool: Path, digest: str):
    import fcntl
    locks = pool / "locks"
    _no_symlinks(locks)
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = locks / (digest + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_nlink != 1:
            raise StorageIntegrityError(f"Pool lock is not independent: {lock}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _verified_clone(source: Path, destination: Path, expected: str, clone) -> None:
    source_state = _state(source)
    clone(source, destination)
    target_state = _state(destination)
    if target_state[:2] == source_state[:2] or target_state[0] != source_state[0]:
        raise StorageIntegrityError("COW clone must have an independent inode on the same filesystem")
    if file_sha256(destination) != expected or _state(source) != source_state:
        raise StorageIntegrityError("COW clone bytes/source identity differ")


def _remove_temporary(directory: Path) -> None:
    try:
        shutil.rmtree(directory)
    except OSError as error:
        # A temporary below base/ would otherwise enter the later byte receipt.
        raise StorageIntegrityError(f"Storage temporary cleanup failed; do not publish: {directory}") from error


def _share_file(source: Path, pool: Path, clone) -> dict:
    source_before = _state(source)
    digest = file_sha256(source)
    canonical = pool / "sha256" / digest[:2] / (digest + ".safetensors")
    with _hash_lock(pool, digest):
        _no_symlinks(canonical)
        canonical.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if canonical.parent.stat().st_dev != source_before[0]:
            raise OSError("Native weight pool and checkpoint must share one filesystem")
        if canonical.exists():
            if file_sha256(canonical) != digest:
                raise StorageIntegrityError(f"Content-addressed pool entry is corrupt: {canonical}")
            if _state(canonical)[:2] == source_before[:2]:
                raise StorageIntegrityError("Saved file must not alias the pool inode")
            temporary = Path(tempfile.mkdtemp(prefix=".native-cow-", dir=source.parent))
            replacement, backup = temporary / "replacement", temporary / "original"
            replaced, preserve_recovery = False, False
            try:
                _verified_clone(canonical, replacement, digest, clone)
                _verified_clone(source, backup, digest, clone)
                metadata = source.stat()
                os.chmod(replacement, stat.S_IMODE(metadata.st_mode))
                os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                with replacement.open("rb") as stream:
                    os.fsync(stream.fileno())
                if _state(source) != source_before:
                    raise StorageIntegrityError("Saved weight changed before COW replacement")
                os.replace(replacement, source)
                replaced = True
                if file_sha256(source) != digest:
                    raise StorageIntegrityError("Installed native weight failed its SHA256 check")
            except BaseException as error:
                if replaced:
                    try:
                        os.replace(backup, source)
                    except Exception as rollback_error:
                        preserve_recovery = True
                        raise StorageIntegrityError(
                            f"Native weight rollback failed ({rollback_error}); recovery retained at {backup}"
                        ) from error
                raise
            finally:
                if not preserve_recovery:
                    _remove_temporary(temporary)
            action = "reused_verified_pool_clone"
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=canonical.parent))
            try:
                staging = temporary / "native.safetensors"
                _verified_clone(source, staging, digest, clone)
                if _state(source) != source_before:
                    raise StorageIntegrityError("Saved weight changed before pool publication")
                # Canonical entries are not save targets. Each output retains its own mode.
                os.chmod(staging, 0o444)
                with staging.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(staging, canonical)
            finally:
                _remove_temporary(temporary)
            action = "published_pool_clone_of_saved_file"
    return {"file": source.name, "sha256": digest, "bytes": source_before[2],
            "pool_relative_path": canonical.relative_to(pool).as_posix(),
            "action": action, "independent_inode": _state(source)[:2] != _state(canonical)[:2]}


def deduplicate_native_weights(base_directory: str | Path, *, pool_directory: str | Path | None = None,
                               clone=None) -> dict:
    """Optimize new saved base/model shards while retaining ordinary-save fallback.

    Default scope is this workspace's runs/. Other export locations are left
    untouched. OSError (e.g. unsupported cloning) retains the valid ordinary save;
    integrity conflicts fail closed. The optional clone argument is for tiny tests.
    """
    if ".." in Path(base_directory).parts or (pool_directory is not None and ".." in Path(pool_directory).parts):
        raise StorageIntegrityError("Checkpoint storage refuses parent path traversal")
    base = Path(base_directory).absolute()
    if pool_directory is None:
        runs = ROOT / "runs"
        if not base.is_relative_to(runs) or base.is_relative_to(runs / ".checkpoint_storage"):
            return {"status": "ordinary_save_outside_runs", "files": []}
        pool = runs / ".checkpoint_storage"
    else:
        pool = Path(pool_directory).absolute()
    if base.name != "base" or pool.name != ".checkpoint_storage" or pool.is_relative_to(base.parent):
        raise StorageIntegrityError("Pool must be named .checkpoint_storage outside the saved checkpoint")
    _no_symlinks(base)
    _no_symlinks(pool)
    if clone is None and platform.system() != "Darwin":
        return {"status": "ordinary_save_no_macos_clone_backend", "files": []}
    clone = clone or clone_file
    files = sorted(path for path in base.iterdir() if NATIVE_WEIGHT.fullmatch(path.name))
    if not files:
        raise StorageIntegrityError("No native saved model safetensors shards found")
    results = []
    for path in files:
        _state(path)
        try:
            results.append(_share_file(path, pool, clone))
        except OSError as error:
            # clonefile never degrades to copying. The preceding ordinary save is
            # already self-contained; do not create another persistent full copy.
            results.append({"file": path.name, "action": "ordinary_save_retained",
                            "reason": str(error)})
    return {"status": "cow_pool_processed", "backend": "macos_clonefile" if clone is clone_file else "injected_test_clone",
            "pool_relative_to_runs": ".checkpoint_storage", "files": results,
            "normal_serialization_performed": True, "pool_gc": "explicit_separate_operation",
            "physical_space_saved_bytes": None}
