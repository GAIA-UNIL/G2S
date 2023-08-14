---
title: Run the server
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
---

# Run the server
Simply run `g2s server` to launch the server, optionally accompanied by any of the below settings. Alternatively, run `g2s enable` to launch the server and make sure it relaunches at start-up when you reboot your machine (and run `g2s disable` to disable this service again).
If you did not install g2s through linuxbrew but chose the manual installation from source instead, launch the server by running `./server` from within the G2S/build/c++-build (or G2S/build/intel-build for the intel version).

ⓘ The server generates logs- and data-files in subfolders /tmp/G2S/logs and /tmp/G2S/data that are saved for debug purposes. They are currently automatically removed at the reboot of the computer, at the launch of the server or after one day. This parameterization can be changed with -kod and -age.

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


