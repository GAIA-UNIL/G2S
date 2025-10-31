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

# Make module callable: g2s(...) acts like g2s.run(...)
class _CallableModule(sys.modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return run(*args, **kwargs)

sys.modules[__name__].__class__ = _CallableModule

# Optional: preserve both names in tab-completion
__all__ = ["run", "g2s"]
