---
title: Installation of interfaces
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 3
---

## Installation of interfaces

<!--Tab Buttons-->
<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'python', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks" onclick="openTab(event, 'matlab', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Matlab.png" alt="Matlab">
  </button>
  <button class="tablinks" onclick="openTab(event, 'R', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Rlogo.svg" alt="R">
  </button>
</div>

<div class="langcontent interface python">

### Python

#### Automatic installation

Simply use `pip install G2S`

#### Manual compilation

1. If needed Python and Numpy: `sudo apt install python3-distutils python3-dev python3-numpy -y`
2. (A C++ compiler with c++17 is required)
3. Go to `build/python-build`
4. Run `python3 setup.py install --user`

#### Check proper interface installation

Simply run `from g2s import g2s; g2s('--version')`
</div>

<div class="langcontent interface matlab">

### MATLAB
<!--Tab Buttons-->
<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'linux', 'os')">
    <img src="{{ site.baseurl }}/assets/images/linux.png" alt="Linux">
  </button>
  <button class="tablinks" onclick="openTab(event, 'macOS', 'os')">
    <img src="{{ site.baseurl }}/assets/images/apple.png" alt="macOS">
  </button>
  <button class="tablinks" onclick="openTab(event, 'Windows', 'os')">
    <img src="{{ site.baseurl }}/assets/images/Windows.png" alt="Windows">
  </button>
</div>

<div class="langcontent os linux">

#### Installation on Ubuntu

1. (A C++ compiler with c++17 is required)
2. Open MATLAB
3. Go to `build/matlab-build`
4. Run `CompileG2S`
5. Add compiled file in the MATLAB path
</div>

<div class="langcontent os macOS">

#### Installation on macOS

1. Open MATLAB
2. Go to `build/build-matlab`
3. Run `CompileG2S`
4. Add compiled file in the MATLAB path
</div>

<div class="langcontent os Windows">

#### Installation on Windows 10

##### Download precompiled interfaces

1. Download [here](https://github.com/GAIA-UNIL/G2S-compiled-interfaces/raw/master/latest/MATLAB/Windows/G2S-latest.win-amd64-matlab.zip).
2. Unzip and add the folder to MATLAB path.

##### Manual compilation

1. (A C++ compiler with c++17 is required)
2. If needed, install [python](https://www.python.org/downloads/) with the option to add it to the Path
3. Open MATLAB
4. Install a compiler with c++17, available [here (2017 or later)](https://visualstudio.microsoft.com/en/downloads)
5. Go to `build/build-matlab`
6. Run `CompileG2S`
7. Add compiled file in the MATLAB path
</div>


##### Check proper interface installation

Simply run `g2s('--version')`

</div>

<div class="langcontent interface R">

## R {#R-lang}

TODO !

</div>