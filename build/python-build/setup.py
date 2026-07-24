#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from importlib.util import module_from_spec, spec_from_file_location

system = platform.system()

# -----------------------------------------------------------------------------
# Basic relative paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT

# -----------------------------------------------------------------------------
# Version handling
# -----------------------------------------------------------------------------
# --- dynamic version from g2s/_version.py ---
VERSION_PATH = Path(__file__).resolve().parent / "g2s" / "_version.py"
spec = spec_from_file_location("g2s._version", VERSION_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load package version from {VERSION_PATH}")
version_module = module_from_spec(spec)
spec.loader.exec_module(version_module)
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
    "src_interfaces/browserTransport.cpp",
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
            dlls = getattr(self, "_g2s_zmq_dlls", [])
            if not dlls:
                raise RuntimeError("The Windows build produced no ZeroMQ DLL to bundle.")
            package_dir = Path(self.build_lib) / "g2s"
            package_dir.mkdir(parents=True, exist_ok=True)
            for dll in dlls:
                dest = package_dir / dll.name
                self.copy_file(str(dll), str(dest))
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
                    "/DG2S_ENABLE_BROWSER_TRANSPORT=1",
                    "/DNOMINMAX",
                    f'/DVERSION="{PACKAGE_VERSION}"',
                    f'/DPYTHON_VERSION="{pyver}"',
                    "/D_CRT_SECURE_NO_WARNINGS",
                    "/DZMQ_NO_PRAGMA_LIB",
                ]
            else:
                ext.extra_compile_args += [
                    cxxflag,
                    "-DG2S_ENABLE_BROWSER_TRANSPORT=1",
                    f'-DVERSION="{PACKAGE_VERSION}"',
                    f'-DPYTHON_VERSION="{pyver}"',
                ]
                ext.extra_link_args += [cxxflag]

            # macOS: arm64 only
            if system == "Darwin":
                os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

            if system == "Windows":
                if "ws2_32" not in ext.libraries:
                    ext.libraries.append("ws2_32")
                libzmq_source = Path(
                    os.environ.get("G2S_LIBZMQ_ROOT", str(ROOT / "libzmq"))
                ).resolve()
                libzmq_root = libzmq_source / "build"
                libdir = libzmq_root / "lib" / "Release"
                dlldir = libzmq_root / "bin" / "Release"

                # MSVC adds toolset and version information to these filenames.
                candidates = sorted(libdir.glob("libzmq*.lib"))
                dlls = sorted(dlldir.glob("libzmq*.dll"))
                if not candidates:
                    raise RuntimeError(
                        f"No libzmq*.lib found under {libdir}. "
                        "Run setup_Win_compile_all.bat before building the wheel."
                    )
                if not dlls:
                    raise RuntimeError(f"No libzmq*.dll found under {dlldir}.")

                libfile = candidates[0]
                print(f"Found ZeroMQ import library: {libfile}")
                print(f"Bundling ZeroMQ DLL: {dlls[0]}")
                ext.include_dirs.append(str(libzmq_source / "include"))
                ext.library_dirs.append(str(libdir))
                ext.libraries.append(libfile.stem)
                self._g2s_zmq_dlls = dlls

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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    ext_package="g2s",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    include_dirs=[],
    package_data={"g2s": ["*.dll"]},
)
