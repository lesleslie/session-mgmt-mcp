from __future__ import annotations

import atexit
import os
import typing as t
from pathlib import Path

_ORIGINAL_ENV: dict[str, str | None] = {}
_TEST_FILENAME = ".sitecustomize-write-test"


def _remember_env(key: str) -> None:
    if key not in _ORIGINAL_ENV:
        _ORIGINAL_ENV[key] = os.environ.get(key)


def _set_env_var(key: str, value: str) -> None:
    _remember_env(key)
    os.environ[key] = value


def _restore_env_vars() -> None:
    for key, previous in _ORIGINAL_ENV.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


atexit.register(_restore_env_vars)


def _ensure_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / _TEST_FILENAME
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return True
    except (OSError, PermissionError):
        return False


def _ensure_cache_paths() -> None:
    candidate_paths: list[Path] = []

    env_cache = os.environ.get("XDG_CACHE_HOME")
    if env_cache:
        candidate_paths.append(Path(env_cache))
    default_cache = Path.home() / ".cache"
    if not env_cache or Path(env_cache) != default_cache:
        candidate_paths.append(default_cache)

    for candidate in candidate_paths:
        if _ensure_directory(candidate):
            return

    fallback = Path.cwd() / ".cache"
    fallback.mkdir(parents=True, exist_ok=True)
    _set_env_var("XDG_CACHE_HOME", str(fallback))

    uv_dir = fallback / "uv"
    uv_dir.mkdir(parents=True, exist_ok=True)
    _set_env_var("UV_CACHE_DIR", str(uv_dir))


def _install_sysconf_fallback() -> None:
    if not hasattr(os, "sysconf"):
        return

    original_sysconf = os.sysconf
    sc_arg_max = os.sysconf_names.get("SC_ARG_MAX", "SC_ARG_MAX")

    def _safe_sysconf(name: t.Any) -> int:
        try:
            return original_sysconf(name)
        except PermissionError:
            if name in ("SC_ARG_MAX", sc_arg_max):
                return 2**17
            raise

    os.sysconf = _safe_sysconf


def _install_crackerjack_lock_fallback() -> None:
    try:
        from crackerjack.config import global_lock_config
    except Exception:
        return

    original_post_init = global_lock_config.GlobalLockConfig.__post_init__

    def _patched_post_init(self: t.Any) -> None:
        try:
            original_post_init(self)
        except PermissionError:
            fallback_dir = Path.cwd() / ".crackerjack" / "locks"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            try:
                fallback_dir.chmod(0o700)
            except PermissionError:
                pass
            self.lock_directory = fallback_dir

    global_lock_config.GlobalLockConfig.__post_init__ = _patched_post_init


_ensure_cache_paths()
_install_sysconf_fallback()
_install_crackerjack_lock_fallback()
