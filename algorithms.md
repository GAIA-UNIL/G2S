---
title: Algorithms
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
---


# Algorithms

- **QuickSampling (QS)** *(aka. Quantile Sampling)* is a general-purpose pixel-based MPS algorithm that is designed to be robust, efficient, and run in constant time. QS was designed to adapt to your problem; it can be used to do (un)conditional simulation, gap filling, or even downscaling, using continuous or categorical variables or a combination of both. The code was developed without restrictions regarding the dimensionality of the data *(e.g. 1D, 2D, 3D, nD)*.
- **Narrow Distribution Selection (NDS)** is an algorithm specifically targeted to simulate spectrally enhanced remote-sensed imagery. It requires an external variable *(for example, a grayscale image)* to control the simulation *(of colors)*.

The framework can be easily extended to handle most codes that use gridded data. Currently, any compiled code or Python code can be handled.
