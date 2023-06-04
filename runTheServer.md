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


