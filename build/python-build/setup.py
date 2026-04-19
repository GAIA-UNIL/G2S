#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from importlib.util import spec_from_file_location

system = platform.system()

# -----------------------------------------------------------------------------
# Basic relative paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).parent            # e.g. .../python/interface/python3
REPO = Path(".")                # go back to repo root relatively
REPO = REPO.resolve()                   # only resolve once for reading files

# -----------------------------------------------------------------------------
# Version handling
# -----------------------------------------------------------------------------
# --- dynamic version from g2s/_version.py ---
VERSION_PATH = Path(__file__).resolve().parent / "g2s" / "_version.py"
spec = spec_from_file_location("g2s._version", VERSION_PATH)
version_module = spec.loader.load_module()
PACKAGE_VERSION = version_module.__version__

if "Test" in os.environ.get("GITHUB_WORKFLOW", ""):
    PACKAGE_VERSION += f".dev{os.environ.get('GITHUB_RUN_NUMBER', '')}"

with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# -----------------------------------------------------------------------------
# JSONCPP linkage
# -----------------------------------------------------------------------------
# Use system libjsoncpp by default. If bundled source exists, build it statically.
extra_cpp = []
extra_libs = ["jsoncpp"]
jsoncpp_src = REPO / "src" / "jsoncpp.cpp"
if jsoncpp_src.is_file():
    print("Using bundled jsoncpp.cpp")
    extra_cpp = ["src/jsoncpp.cpp"]
    extra_libs = []


# Only link zlib on Linux / macOS
if system in ("Linux", "Darwin"):
    extra_libs.append("z")

# -----------------------------------------------------------------------------
# Source files (now trivially relative)
# -----------------------------------------------------------------------------
sources = [
    "src_interfaces/python3_interface.cpp",
    "src/DataImage.cpp",
] + extra_cpp

print("sources (relative):", sources)


def _collect_pyzmq_paths():
    include_dirs = []
    library_dirs = []
    try:
        import zmq
    except Exception as exc:
        print(f"pyzmq import unavailable during build: {exc}")
        return include_dirs, library_dirs

    try:
        get_includes = getattr(zmq, "get_includes", None)
        if callable(get_includes):
            include_dirs.extend([str(Path(p)) for p in get_includes() if p])
    except Exception as exc:
        print(f"pyzmq include path detection failed: {exc}")

    try:
        get_library_dirs = getattr(zmq, "get_library_dirs", None)
        if callable(get_library_dirs):
            library_dirs.extend([str(Path(p)) for p in get_library_dirs() if p])
    except Exception as exc:
        print(f"pyzmq library path detection failed: {exc}")

    # Fallback probes used by some pyzmq wheel layouts.
    zmq_pkg_dir = Path(zmq.__file__).resolve().parent
    for candidate in (zmq_pkg_dir / "include", zmq_pkg_dir.parent / "include"):
        if candidate.is_dir():
            include_dirs.append(str(candidate))

    # Preserve order and drop duplicates.
    include_dirs = list(dict.fromkeys(include_dirs))
    library_dirs = list(dict.fromkeys(library_dirs))
    return include_dirs, library_dirs

# -----------------------------------------------------------------------------
# Custom build_ext (adds NumPy includes and platform flags)
# -----------------------------------------------------------------------------
class build_ext(_build_ext):
    def run(self):
        super().run()

        # Copy DLLs after extension build (Windows only)
        if system == "Windows":
            dlldir = Path("libzmq") / "build" / "bin" / "Release"
            if dlldir.exists():
                for dll in dlldir.glob("libzmq*.dll"):
                    dest = Path(self.build_lib) / "g2s" / dll.name
                    self.copy_file(dll, dest)
                    print(f"Copied {dll.name} to {dest}")

    def build_extensions(self):
        import numpy as np

        pyzmq_include_dirs, pyzmq_library_dirs = _collect_pyzmq_paths()

        for ext in self.extensions:
            ext.include_dirs.append(np.get_include())

            # Prefer pyzmq-provided headers first (contains zmq.h in isolated builds).
            ext.include_dirs += pyzmq_include_dirs
            ext.include_dirs += [
                str(REPO / "include"),
                str(REPO / "include_interfaces"),
                "/usr/include",
                "/usr/include/jsoncpp",
                "/opt/local/include",
                "/opt/homebrew/include",
            ]

            ext.library_dirs += pyzmq_library_dirs
            ext.library_dirs += [
                "/usr/lib",
                "/opt/local/lib",
                "/opt/homebrew/lib",
            ]

            # C++20 flags
            if system == "Windows":
                cxxflag = "/std:c++20"
            else:
                cxxflag = "-std=c++20"

            pyver = platform.python_version()
            if system == "Windows":
                ext.extra_compile_args += [
                    cxxflag,
                    "-DNOMINMAX",
                    f'/DVERSION="{PACKAGE_VERSION}"',
                    f'/DPYTHON_VERSION="{pyver}"',
                    "/D_CRT_SECURE_NO_WARNINGS",
                    "/D_USE_MATH_DEFINES",  # 👈 Add this line
                    "/D_CRT_SECURE_NO_WARNINGS",  # 👈 Optional: silence fopen/fscanf warnings
                    "/DZMQ_NO_PRAGMA_LIB"
                ]
            else:
                ext.extra_compile_args += [
                    cxxflag,
                    f'-DVERSION="{PACKAGE_VERSION}"',
                    f'-DPYTHON_VERSION="{pyver}"',
                ]
                ext.extra_link_args += [cxxflag]

            # macOS: arm64 only
            if system == "Darwin":
                os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")
                ext.extra_compile_args += ["-arch", "arm64"]
                ext.extra_link_args += ["-arch", "arm64"]

            if system == "Windows":
                libzmq_root = Path("libzmq") / "build"
                libdir = libzmq_root / "lib" / "Release"
                dlldir = libzmq_root / "bin" / "Release"

                # detect actual library name (MSVC adds version tags)
                candidates = list(libdir.glob("libzmq*.lib"))
                if candidates:
                    libfile = candidates[0]
                    libname = libfile.stem
                    print(f"Found ZeroMQ lib: {libfile.name}")

                    for ext in self.extensions:
                        ext.include_dirs += [str(Path("libzmq") / "include")]
                        ext.library_dirs += [str(libdir)]
                        ext.libraries += [libname]
                else:
                    print("No libzmq*.lib found under libzmq/build/lib/Release — build likely incomplete.")

                # copy DLLs for wheel packaging
                if dlldir.exists():
                    self.copy_dlls = list(dlldir.glob("libzmq*.dll"))
                    if self.copy_dlls:
                        print(f"Bundling {len(self.copy_dlls)} ZeroMQ DLL(s) into wheel.")
                else:
                    print("No bin/Release folder found — DLL will not be packaged.")

            if not any((Path(include_dir) / "zmq.h").is_file() for include_dir in ext.include_dirs):
                raise RuntimeError(
                    "Unable to locate zmq.h. Install libzmq headers (e.g. libzmq3-dev/zeromq-devel) "
                    "or ensure pyzmq headers are available in the build environment."
                )

        super().build_extensions()

# -----------------------------------------------------------------------------
# Extension definition
# -----------------------------------------------------------------------------
ext = Extension(
    "g2s",
    sources=sources,
    language="c++",
    libraries= extra_libs,
)

# -----------------------------------------------------------------------------
# setup()
# -----------------------------------------------------------------------------
setup(
    name="G2S",
    version=PACKAGE_VERSION,
    description="G2S Python interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mathieu Gravey",
    author_email="g2s@mgravey.com",
    url="https://github.com/GAIA-UNIL/G2S",
    packages=["g2s"],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
    ],
    ext_package="g2s",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    include_dirs=[],
    data_files=[
        ("g2s", [str(p) for p in getattr(build_ext, "copy_dlls", [])])
    ],
)
