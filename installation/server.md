---
title: Installation of the server
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 3
---


## Installation of the server {#installServer}
<!--Tab Buttons-->
<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'linux','os')">
    <img src="{{ site.baseurl }}/assets/images/linux.png" alt="Linux">
  </button>
  <button class="tablinks" onclick="openTab(event, 'macOS','os')">
    <img src="{{ site.baseurl }}/assets/images/apple.png" alt="macOS">
  </button>
  <button class="tablinks" onclick="openTab(event, 'Windows','os')">
    <img src="{{ site.baseurl }}/assets/images/Windows.png" alt="Windows">
  </button>
  <a href="/installation/cluster.html">
    <button class="tablinks">
      <img src="{{ site.baseurl }}/assets/images/cloudGear.svg" alt="Windows">
    </button>
  </a>
</div>

<div class="langcontent os linux">

### Installation of the server on Ubuntu

1. Install linuxbrew `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Run `brew install open-geocomputing/brew/g2s`
3. Use the `g2s`command 

</div>

<div class="langcontent os macOS">

### Installation of the server on macOS

1. Install homebrew if not already done `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Run `brew install open-geocomputing/brew/g2s`
3. Use the `g2s`command 

</div>

<div class="langcontent os Windows">

### Installation of the server on Windows 10

1. Check that the latest updates of Windows are installed
2. Install WSL following [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10), and select a Linux distribution (we recommend choosing Ubuntu for beginners).
3. Install linuxbrew `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
4. Run `brew install open-geocomputing/brew/g2s`
5. Use the `g2s`command 
</div>


<div class="langcontent os linux">

### Installation of the server manually from source on Ubuntu

#### Automatic with GCC on Ubuntu

1. First clone the code from this GitHub: `git clone https://github.com/GAIA-UNIL/G2S.git`
2. Then run `build/c++-build/install_needs_W_VM.sh`

#### Manual (including Intel C++ Compiler)

1. First clone the code from this GitHub: `git clone https://github.com/GAIA-UNIL/G2S.git`
2. Basics for compiling are needed (e.g. on Ubuntu: build-essential).
3. The following packages are required: ZMQ, JsonnCpp and zlib for G2S; fftw3 for QS and NDS.
4. To install them on Ubuntu: `sudo apt install build-essential libzmq3-dev libjsoncpp-dev zlib1g-dev libfftw3-dev libcurl4-openssl-dev -y` (libcurl4-openssl-dev is optional).
5. The C++ extension for ZMQ is required too, and can be installed via: `wget "https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp" -O ./include/zmq.hpp`
6. Go to the build subfolder.
7. Run: `make -j`, if the Intel C++ compiler is installed, the adapted version will be compiled too. The Intel compiler can be downloaded freely in many cases: [here](https://software.intel.com/en-us/qualify-for-free-software).
   To manually select between GCC or Intel compiler use `make c++ -j` or `make intel -j`, respectively.

</div>

<div class="langcontent os macOS">

### Installation of the server manually from source on macOS

#### using Homebrew (now recommended)

1. First clone the code from this GitHub: `git clone https://github.com/GAIA-UNIL/G2S.git`
2. Install the package manager [Homebrew](https://brew.sh/) (if not already done)
3. The following packages are required: ZMQ, JsonnCpp and zlib for G2S. fftw3 for QS and NDS.
4. To install them with Homebrew: `brew install fftw zmq jsoncpp cppzmq curl` Note: curl is optional.
5. Go to the `build` subfolder.
6. Run: `make -j`, if the Intel C++ compiler is installed the adapted version will be compiled too. The Intel compiler can be downloaded freely in many cases: [here](https://software.intel.com/en-us/qualify-for-free-software). Obviously, Intel compiler is only for Intel CPUs 😁. To manually select between GCC or Intel compiler use `make c++ -j` or `make intel -j`, respectively.

#### using MacPort (now deprecated)

1. First clone the code from this GitHub: `git clone https://github.com/GAIA-UNIL/G2S.git`
2. Install the package manager [MacPort](https://www.macports.org/install.php) (if not already done)
3. The following packages are required: ZMQ, JsonnCpp and zlib for G2S. fftw3 for QS and NDS.
4. To install them with macPort: `sudo port install zmq-devel jsoncpp-devel zlib cppzmq-devel fftw-3 fftw-3-single curl` Note: curl is optional.
5. Go to the `build` subfolder.
6. Run: `make -j`, if the Intel C++ compiler is installed the adapted version will be compiled too. The Intel compiler can be downloaded freely in many cases: [here](https://software.intel.com/en-us/qualify-for-free-software). 
To manually select between GCC or Intel compiler use `make c++ -j` or `make intel -j`, respectively.

</div>

<div class="langcontent os Windows">

### Installation of the server manually from source on Windows 10

1. Check that the latest updates of Windows are installed
2. Install WSL following [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10), and select a Linux distribution (we recommend choosing Ubuntu for beginners).
3. Clone the code from this GitHub repository: https://github.com/GAIA-UNIL/G2S.git (or download and unzip the zip-file on the top right of this page).
4. In Windows, go to the directory `build/c++-build` and run/double-click `install.bat`.

</div>
