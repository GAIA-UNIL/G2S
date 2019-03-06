# G2S

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Brief overview

**G2S** is composed of 2 parts:
- the first one is a server that manages computations and can be compiled for each hardware to obtain optimal performance.
- the second part is composed of different interfaces that communicate with the server through ZeroMQ. Interfaces can be added for each software, in the same way G2S can be extended for any other algorithm.

Currently the **G2S** interface is available for *MATLAB* and *Python*. **G2S** is provided with both algorithm **QS** (Quantile Sampling) and **NDS** (Narrow Distribution Selection).

**G2S** is currently only available for *UNIX*-based system, *Linux* and *macOS*. A solution for *Windows 10* is provided using *WSL* (Windows Subsystem for Linux). However, for previous *Windows* versions, the only solution currently available is to install a *Linux* system manually inside a virtual machine. 

## Installation
The installation is in two parts:
- Installation of the server
- Installation of interfaces

### Installation of the server

#### Ubuntu (and to adapt for other Linux distribution)

##### Automatic with *GCC*:
1. First clone the code from this GitHub.
2. run `build/c++-build/install_needs_W_VM.sh`

##### Manual (included *Intel C++ Compiler*):
1. First clone the code from this GitHub.
2. Basics for compiling are needed, (e.g. on Ubuntu: build-essential).
3. The following packages are required: ZMQ, JsonnCpp and zlib for G2S. fftw3 for QS and NDS.  
To install them on Ubuntu:
`sudo apt install build-essential libzmq3-dev libjsoncpp-dev zlib1g-dev libfftw3-dev libcurl4-openssl-dev –y` (libcurl4-openssl-dev is optional)
4. The C++ extension for ZMQ is required too, and can be installed via:
`sudo wget "https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp" -O /usr/include/zmq.hpp`
5. Go to the `build` subfolder.
6. Run: `make c++ -j` or `make intel -j` if the *Intel C++ compiler* is installed.
	The intel compiler can be downloaded freely in many cases: [here](https://software.intel.com/en-us/qualify-for-free-software)

#### macOS
1. First clone the code from this GitHub.
2. Install a package manager like [macPort](https://www.macports.org/install.php) (or for expert with [Homebrew](https://brew.sh) )
3. The following packages are required: ZMQ, JsonnCpp and zlib for G2S. fftw3 for QS and NDS.  
To install them with macPort: (to adapt for brew)
`sudo port install zmq-devel jsoncpp-devel zlib cppzmq-devel fftw-3 fftw-3-single curl` (curl is optional)
5. Go to the `build` subfolder.
6. Run: `make c++ -j` or `make intel -j` if the *Intel C++ compiler* is installed.
	The intel compiler can be downloaded freely in many cases: [here](https://software.intel.com/en-us/qualify-for-free-software)

#### Windows 10
1. Check that the last updates of Windows are installed
2. Install WSL follwing instruction [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
3. Go to the directory `build/c++-build`
4. Run `install.bat`

### Installation of interfaces

#### MATLAB
* Solution 1:
1. Download precompiled interfaces [here](http://wp.unil.ch/gaia/files/2018/08/Prebuild-G2S-MATLAB-Interfaces.zip)
2. Add downloaded file in the MATLAB path
* Solution 2 (for Linux and macOS):
1. Open MATLAB
2. Go to `build/build-matlab`
3. Run `CompileG2S`	
4. Add compiled file in the MATLAB path
* Solution 3, for windows (all versions)
0. If needed install [python](https://www.python.org/downloads/) with the option to add it to the Path 
1. Open MATLAB
2. Install a compiler, available [here](https://mathworks.com/matlabcentral/fileexchange/52848-matlab-support-for-mingw-w64-c-c-compiler)  
3. Go to `build/build-matlab`
4. Run `CompileG2S`	
5. Add compiled file in the MATLAB path

#### Python (Python 3) /Numpy
* Solution 1, for windows (all versions)
0. If needed, install [python](https://www.python.org/downloads/) with the option to add it to the Path  
Install [Visual C++ Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=48145) that is needed for scipy (don't ask me why)
1. Install needed package `pip install pillow scipy scikit-misc matplotlib`
2. Download precompiled interfaces [here](http://wp.unil.ch/gaia/files/2018/09/G2S-python3-1.zip)
* Solution 2
0. If needed Python and Numpy  
Ubuntu: `sudo apt install python3-distutils python3-dev python3-numpy –y`  
macOS: `sudo port install python37 py37-numpy`
1. Go to `build/python-build`
2. Run `sudo python3 setup.py install`

## Run the server

:warning: The utilization of the server generate logs- and data-files, in subfolder `build/build-*/logs` and `build/build-*/data` that are originally saved for debug purpose, and are currently automatically removed only at the launch of the server or after one day. This parameterization can be changed with `-kod` and `-age`

#### Ubuntu / macOS
Run `./server` in `build/c++-build` or `build/intel-build`, respectively for the standard or Intel version  
Options:

Flag | Description
-- | --
-d | run as daemon
-To n | shutdown the server if there is no activity after n second, 0 for ∞ (default : 0)
-p | select a specific port for the server (default: 8128)
-kod | keep old data, if the flag is not present all files from previous runs are erased at launch
-age | set the time in second after files need to be removed (default: 86400 s = 1d)
-mT | single job at the time (experimental)
-fM | Run as function, without fork (experimental)

#### Windows 10
It is possible to run the server with `runServer.bat` or `runDaemon.bat` as daemon available in `build/c++-build`

## Use the interfaces
To do any computation a call to **g2s** is needed:
Parameter do **_NOT_** have a specific order.

#### In MATLAB
``` MATLAB
data=g2s(…)		% principal output, the simulation
[data, t]=g2s(…)	% the simulation and the computation time
[data, …,t]=g2s(…)	% the simulation, other outputs map and the computation time
```

#### In Python
```Python
from g2s import run as g2s
data=g2s(…)	# it returns an array that contains all the output maps and the computing duration
```

Each call to **G2S** is composed of parameters of G2S and of the name of the algorithm used.

#### Options

Flag | Description
-- | --
-a | the algorithm, it can be ‘qs’, ‘nds’, ‘ds-l’
-sa | address of the computation server (default: localhost, the server is local)<br/>Nice when we have a powerful machine dedicated for computation
-silent | don’t display the progression, nice for scripts
-noTO | deactivate timeout on ZMQ communication, useful for slow network (e.g. trough internet) 
-shutdown | shutdown the server, useful at the end of scripts

The following options represent the *Async* mode, check example in *MATLAB* and *Python* directory:

Flag | Description
-- | --
-submitOnly | submit a job
-statusOnly | check progression
-waitAndDownload | download the result

#### QS options

| Mandatory | Flag | Description
| -- | -- | --
|:heavy_check_mark: | -ti | Training images (1+)
|:heavy_check_mark: | -di | Destination image (1) (aka simulation grid)
|:heavy_check_mark: | -dt | Data type, array of to specify the type of each variable, 0 &rightarrow; continuous and 1 &rightarrow; categorical
| | -sp | simulation path, image of the position in the path, small values first, -∞ are not simulated
| | -ki | image of weighting kernel 
:heavy_check_mark: | -k | the number of best candidates to consider ∈[1 ∞]
| | -f | f=1/k equivalent to f of DS with a threshold to 0
:heavy_check_mark: | -n | the N closest neighbors to consider:<br> - single value for vectors neighbors<br> - one for each variable 
| | -s | random seed value
| | -j | to run in parallel (if specified), to use as follows ‘-j’, N1, N2, N3<br/>- N1 threads used to parallelize the path (path-level) Default: the maximum number of threads available.<br/>- N2 threads used to parallelize over training images (node-level), work only if many TIs are available. Default: 1<br/>- N3 threads used to parallelize FFTs (path-level). Default: 1<br/>- Favorize N1 and N2 over N3, N1 is usually more efficient than N2, but require more memory.
| | -W_GPU | use integrated GPU if available
| | -fs | full simulation, run the simulation of each pixel independentantly, i.e. dosen't simulate all the vector at the time (experimental)
| | -nV | no Verbatim (experimental)

##### Return 
- Simulation
- Encoded position of the original value for each pixel
- Duration

#### NDS options

| Mandatory | Flag | Description
| -- | -- | --
|:heavy_check_mark: | -ti | Training images (one or more images)
|:heavy_check_mark: | -di | Destination image (one image, aka simulation grid)
|:heavy_check_mark: | -dt | Data type, array of to specify the type of each variable, 0 &rightarrow; continuous and 1 &rightarrow; categorical
| | -ki | image of weighting kernel 
|:heavy_check_mark: | -k | the number of best candidates to consider to compute the narrowness ∈[5 ∞]
|:heavy_check_mark: | -n | the N closest neighbors to consider
| | -nw | narrowness range  0&rightarrow; max-min, 1 &rightarrow; median, default IQR &rightarrow; 0.5
| | -nwv | number of variables to consider in the narrowness, (start from the end), default: 1
| | -cs | chunk size, the number of pixels to simulate at the same time, at each iteration, default: 1
| | -uds | area to update around each simulated pixel, the M closest pixel default: 10
| | -mp | partial simulation, 0 &rightarrow; empty, 1 &rightarrow; 100%
| | -s | seed value
| | -j | to run in parallel (if specified), to use as follows ‘-j’, N1, N2, N3<br/>- N1 threads used to parallelize the path (path-level) Default: the maximum number of threads available.<br/>- N2 threads used to parallelize over training images (node-level), work only if many TIs are available. Default: 1<br/>- N3 threads used to parallelize FFTs (path-level). Default: 1<br/>- Favorize N1 and N2 over N3, N1 is usually more efficient than N2, but require more memory.
| | -W_GPU | use integrated GPU if available
| | -nV | no Verbatim (experimental)

##### Return 
- Simulation
- Encoded position of the original value for each pixel
- Simulation path
- Narrowness, for each pixel the simulation narrowness
- Duration

## Example

#### Matlab
```MATLAB
% Minimum
data=g2s('-a','qs','-ti',source,'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50);

% Complete
[data,indexes,t]=g2s('-sa',serverAddress,'-a','qs','-ti',source,rot90(source,1),'-di',destination,'-dt',zeros(1,1),'-k',1.5,'-n',50,'-s',100,'-j',4,1,1,'-ki',kernel,'-sp',reshape(randperm(numel(destination)),size(destination)),'-silent');
```

#### Python
```Python
# Minimum
data=g2s('-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50);

# Complete
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-j',4,1,1,'-ki',kernel,'-sp',numpy.random.permutation(numpy.size(destination)).astype(float).reshape(numpy.shape(destination)),'-silent');
```

## Benchmarking

This code can be used for benchmarking, the code needs to run natively on macOS or on Linux using the Intel Compiler with MKL library, the version needs to be informed, and the time needs to be the time reported by the algorithm (that is the true execution time without taking into account interfaces overhead).

When benchmarking, the code should NOT be used inside a Virtual Machine or truth WSL on Windows 10.
