# g2s/__init__.py
import os
import sys

import ctypes
import zmq.backend.cython._zmq as _zmq

# Export ZMQ symbols globally for g2s.so
try:
    ctypes.CDLL(_zmq.__file__, mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass


_extra_dll_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib', 'site-packages', 'g2s')
if sys.platform == 'win32' and os.path.isdir(_extra_dll_dir):
    if sys.version_info >= (3, 8):
        try:
            os.add_dll_directory(_extra_dll_dir)
        except Exception:
            pass
    else:
        os.environ.setdefault('PATH', '')
        os.environ['PATH'] += os.pathsep + _extra_dll_dir

from .g2s import run
from .g2s import run as g2s


def schema_to_legacy(result):
    if not isinstance(result, dict):
        return result

    legacy = []
    output_keys = []
    if "simulation" in result:
        output_keys.append("simulation")
    artifact_keys = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
    for key in artifact_keys.keys():
        if key in {"log", "warning", "error", "progress", "meta", "simulation"}:
            continue
        if key in result:
            output_keys.append(key)
    for key in sorted(set(output_keys), key=output_keys.index):
        legacy.append(result[key])
    if "time" in result:
        legacy.append(result["time"])
    meta = {}
    for key, value in result.items():
        if key in {"simulation", "time", "job_id", "status", "progress", "artifacts", "error", "warnings"}:
            continue
        if key in artifact_keys:
            continue
        meta[key] = value
    if meta:
        legacy.append(meta)
    if "progress" in result:
        legacy.append(result["progress"])
    if "job_id" in result:
        legacy.append(result["job_id"])
    if len(legacy) == 1:
        return legacy[0]
    return tuple(legacy)

# Make module callable: g2s(...) acts like g2s.run(...)
class _CallableModule(sys.modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return run(*args, **kwargs)

sys.modules[__name__].__class__ = _CallableModule

# Optional: preserve both names in tab-completion
__all__ = ["run", "g2s", "schema_to_legacy"]
