#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import platform
import shutil
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
REPO = Path("../..")                # go back to repo root relatively
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
# JSONCPP detection
# -----------------------------------------------------------------------------
extra_cpp = []
extra_libs = ["jsoncpp", "z"]
jsoncpp_src = REPO / "jsoncpp-master" / "dist" / "jsoncpp.cpp"
if jsoncpp_src.is_file():
    extra_cpp = ["../../jsoncpp-master/dist/jsoncpp.cpp"]
    extra_libs = ["z"]

# -----------------------------------------------------------------------------
# ZeroMQ linkage
# -----------------------------------------------------------------------------
extra_objects = []
libzmq_libs = ["zmq"]
static_zmq_path = os.getenv("STATIC_ZMQ_PATH")
if static_zmq_path:
    libzmq_libs = []
    extra_objects = [static_zmq_path]

# -----------------------------------------------------------------------------
# Homebrew include/lib paths (macOS)
# -----------------------------------------------------------------------------
extra_include_dirs = []
extra_library_dirs = []
if shutil.which("brew"):
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix"]).decode().strip()
        print("brew prefix:", brew_prefix)
        extra_include_dirs.append(os.path.join(brew_prefix, "include"))
        extra_library_dirs.append(os.path.join(brew_prefix, "lib"))
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Source files (now trivially relative)
# -----------------------------------------------------------------------------
sources = [
    "../../src_interfaces/python3_interface.cpp",
    "../../src/DataImage.cpp",
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
            ] + extra_include_dirs

            ext.library_dirs += [
                "/usr/lib",
                "/opt/local/lib",
                "/opt/homebrew/lib",
            ] + extra_library_dirs

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
                ext.extra_link_args += [cxxflag,"-g", "-O0"]

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
    extra_compile_args=[],
    extra_link_args=[],
    extra_objects=extra_objects,
    include_dirs=[],   # filled later
    libraries=libzmq_libs + extra_libs,
    library_dirs=[],
)

# -----------------------------------------------------------------------------
# Windows DLL bundling
# -----------------------------------------------------------------------------
data_files = []
if system == "Windows" and not static_zmq_path:
    zmqBuildDir = ROOT / "libzmq" / "action_build"
    bin_dir = zmqBuildDir / "bin" / "Release"
    if bin_dir.is_dir():
        dlls = [
            str(bin_dir / x)
            for x in os.listdir(bin_dir)
            if x.endswith(".dll") and "libzmq" in x and "mt-s" not in x
        ]
        if dlls:
            data_files = [("lib\\site-packages\\g2s", dlls)]

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
    data_files=data_files,
)
