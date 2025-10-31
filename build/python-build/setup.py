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

# -----------------------------------------------------------------------------
# Custom build_ext (adds NumPy includes and platform flags)
# -----------------------------------------------------------------------------
class build_ext(_build_ext):
    def build_extensions(self):
        import numpy as np

        for ext in self.extensions:
            ext.include_dirs.append(np.get_include())

            ext.include_dirs += [
                str(REPO / "include"),
                str(REPO / "include_interfaces"),
                "/usr/include",
                "/usr/include/jsoncpp",
                "/opt/local/include",
                "/opt/homebrew/include",
            ]

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
    include_dirs=[]
)
