"""
Utility functions for PLAST, including GPU check, distance calculations, and OS page cache warming.
"""

import subprocess
import os
import hashlib
import mmap
import ctypes
import math
import fcntl
from pathlib import Path
from typing import Any, List, Union, Optional
import numpy as np
import numba


def is_gpu_available() -> bool:
    """
    Check if a GPU is available on the system.

    :returns: True if a GPU is available, False otherwise.
    :rtype: bool
    """
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except subprocess.CalledProcessError:
        return False


@numba.njit(fastmath=True)
def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    :param x: First vector.
    :type x: np.ndarray
    :param y: Second vector.
    :type y: np.ndarray
    :returns: Euclidean distance.
    :rtype: float
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    return d


@numba.jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray) -> float:
    """
    copied from https://github.com/pranaychandekar/numba_cosine
    """
    assert u.shape[0] == v.shape[0]
    uv = 0.0
    uu = 0.0
    vv = 0.0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1.0
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta


def warm_page_cache(
    files: List[Union[str, Path]],
    threshold: float = 0.9,
    state_dir: Optional[Union[str, Path]] = None,
    logger: Optional[Any] = None,
) -> None:
    """
    Warm OS page cache for given files only if not sufficiently cached.

    :param files: List of file paths (str or Path). Non-existent files are ignored.
    :type files: list[str or Path]
    :param threshold: Target cached fraction in [0,1] before skipping warm.
    :type threshold: float
    :param state_dir: Directory to place lock/sentinel files (default: $XDG_RUNTIME_DIR or /tmp).
    :type state_dir: str or Path, optional
    :param logger: Optional logger with .debug().
    :type logger: Any, optional
    :returns: None
    """
    paths = [Path(p) for p in files if p is not None]
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        return

    def _log_debug(msg: str) -> None:
        if logger and hasattr(logger, "debug"):
            logger.debug(msg)

    def _file_cached_fraction(path: Path) -> float:
        try:
            sz = path.stat().st_size
            if sz == 0:
                return 1.0
            pagesize = mmap.PAGESIZE
            npages = math.ceil(sz / pagesize)

            fd = os.open(str(path), os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, sz, access=mmap.ACCESS_COPY)
                try:
                    buf = (ctypes.c_ubyte * npages)()
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    libc.mincore.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_ubyte),
                    ]
                    libc.mincore.restype = ctypes.c_int
                    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                    ret = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(sz), buf)
                    if ret != 0:
                        return 0.0
                    resident = sum(b & 1 for b in buf)
                    return resident / npages
                finally:
                    mm.close()
            finally:
                os.close(fd)
        except (OSError, ValueError):
            return 0.0

    def _db_cached_fraction(_paths: List[Path]) -> float:
        total = 0
        acc = 0.0
        for p in _paths:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            frac = _file_cached_fraction(p)
            acc += frac * size
            total += size
        return (acc / total) if total > 0 else 1.0

    def _posix_prefetch(fd: int) -> None:
        """
        Best-effort posix_fadvise(WILLNEED). Not fatal if unavailable.

        :param fd: File descriptor.
        :type fd: int
        :returns: None
        """
        try:
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        except OSError:
            pass

    def _warm(_paths: List[Path]) -> None:
        """
        Read files in chunks to warm the OS page cache.

        :param _paths: List of file paths.
        :type _paths: list[Path]
        :returns: None
        """
        chunk = 8 * 1024 * 1024  # 8MB
        for p in _paths:
            try:
                with open(p, "rb", buffering=1024 * 1024) as f:
                    _posix_prefetch(f.fileno())
                    while f.read(chunk):
                        pass
            except (OSError, ValueError):
                continue

    parts = []
    for p in paths:
        try:
            st = p.stat()
            parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            continue
    sig = (
        hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest() if parts else "nosig"
    )

    if state_dir is None:
        state_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    else:
        state_dir = Path(state_dir)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    sentinel = state_dir / f".warm_cache_done.{sig}"
    if sentinel.exists():
        frac = _db_cached_fraction(paths)
        _log_debug(f"warm_cache: sentinel exists; cached ~{frac * 100:.1f}%")
        if frac >= threshold:
            return

    lock_path = state_dir / ".warm_cache.lock"

    try:
        with open(lock_path, "w", encoding="utf-8") as lockf:
            try:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return

            frac2 = _db_cached_fraction(paths)
            _log_debug(f"warm_cache: cached (post-lock) ~{frac2 * 100:.1f}%")
            if frac2 < threshold:
                _log_debug("warm_cache: warming files into page cache...")
                _warm(paths)
                try:
                    sentinel.touch()
                except OSError:
                    pass

            try:
                fcntl.flock(lockf, fcntl.LOCK_UN)
            except OSError:
                pass
    except OSError:
        frac3 = _db_cached_fraction(paths)
        if frac3 < threshold:
            _warm(paths)
