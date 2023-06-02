---
title: Run the server
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
---

# Run the server
Simply run `g2s server`, logs and data files are store in /tmp/G2S, and automatically removed at the 

ⓘ The server generates logs- and data-files in subfolders /tmp/G2S/logs and /tmp/G2S/data that are originally saved for debug purpose, 
and are currently automatically removed at teh reboot of teh computer, at the launch of the server or after one day. This parameterization can be changed with -kod and -age.

<!-- ⚠️The server generates logs- and data-files in subfolders build/build-*/logs and build/build-*/data that are originally saved for debug purpose, 
and are currently automatically removed only at the launch of the server or after one day. This parameterization can be changed with -kod and -age.


<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'linux','os')">
    <img src="/assets/images/linux.png" alt="Linux">
  </button>
  <button class="tablinks" onclick="openTab(event, 'macOS','os')">
    <img src="/assets/images/apple.png" alt="macOS">
  </button>
  <button class="tablinks" onclick="openTab(event, 'Windows','os')">
    <img src="/assets/images/Windows.png" alt="Windows">
  </button>
</div>

<div class="langcontent os linux">
### Run the server on Ubuntu
Run `./server` in `build/c++-build` or `build/intel-build`, respectively for the standard or Intel version. 
Running the server from a different directory (e.g. `./c++-build/server`) is currently not supported.
</div>

<div class="langcontent os macOS">
### Run the server on macOS
Run `./server` in `build/c++-build` or `build/intel-build`, respectively for the standard or Intel version. 
Running the server from a different directory (e.g. `./c++-build/server`) is currently not supported.
</div>

<div class="langcontent os Windows">
### Run the server on Windows 10+
It is possible to run the server with `runServer.bat` or `runDaemon.bat` as daemon available in `build/c++-build`.
</div>
 -->
### Parameterization

| Flag | Description |
| --- | --- |
| `-d` | Run as daemon |
| `-To n` | Shutdown the server if there is no activity after n seconds, 0 for ∞ (default: 0) |
| `-p` | Select a specific port for the server (default: 8128) |
| `-kod` | Keep old data, if the flag is not present all files from previous runs are erased at launch |
| `-age` | Set the time in second after files need to be removed (default: 86400 s = 1d) |

#### Advanced parameters

| Flag | Description |
| --- | --- |
| `-mT` | Single job at a time (excludes the use of -after, only use if you're sure to be the only one accessing the server) |
| `-fM` | Run as a function, without fork |
| `-maxCJ n` | Limit the maximum number of jobs running in parallel. |


