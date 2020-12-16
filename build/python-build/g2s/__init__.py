import os
import sys

if sys.platform == 'win32' and os.path.isdir(extra_dll_dir):
	extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')
	if sys.version_info >= (3, 8):
		os.add_dll_directory(extra_dll_dir)
	else:
		os.environ.setdefault('PATH', '')
		os.environ['PATH'] += os.pathsep + extra_dll_dir

from .g2s import run
from .g2s import run as g2s