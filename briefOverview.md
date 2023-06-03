---
title: Brief overview
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
permalink: BriefOverview.html
redirect_from:
  - /
---


# Brief overview {#briefOverview}

The **GeoStatistical Server (G2S)** is a framework that allows you to use state-of-the-art **Multiple Point Statistics (MPS)** algorithms to run stochastic simulations.

G2S is designed to run simulations in a generic way, independently of the code used or language it is written in. For example, it enables you to run a *C/C++* simulation code using *Python*, or *Python* using *MATLAB* (or any other combination).

Currently, the framework is provided with:

- **QuickSampling (QS)** *(aka. Quantile Sampling)* is a general-purpose pixel-based MPS algorithm that is designed to be robust, efficient, and run in constant time. QS was designed to adapt to your problem; it can be used to do (un)conditional simulation, gap filling, or even downscaling, using continuous or categorical variables or a combination of both. The code was developed without restrictions regarding the dimensionality of the data *(e.g. 1D, 2D, 3D, nD)*.
  - A journal article published in Geoscientific Model Development describes the method [here](https://gmd.copernicus.org/articles/13/2611/2020/).
- **Narrow Distribution Selection (NDS)** is an algorithm specifically targeted to simulate spectrally enhanced remote-sensed imagery. It requires an external variable *(for example, a grayscale image)* to control the simulation *(of colors)*.
  - A paper describing it is available here: [Analogue-based colorization of remote sensing images using textural information](https://doi.org/10.1016/j.isprsjprs.2018.11.003).

The framework can be easily extended to handle most codes that use gridded data. Currently, any compiled code or Python code can be handled.

For a hands-on introduction to MPS *(PPT slides, Colab Notebook & recorded tutorial)*, follow [this link](https://github.com/GAIA-UNIL/Short-course-MPS).