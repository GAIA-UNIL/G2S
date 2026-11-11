---
title: Resources
author:
  - Mathieu Gravey
date: 2026-11-11
description: Teaching resources for G2S, including animations of QuickSampling, Direct Sampling and SNESIM that show how a sequential simulation works, node by node.
toc-depth: 2
permalink: /resources.html
video_base: https://github.com/GAIA-UNIL/G2S/releases/download/animations-v1
animations:
  - name: qs_strebelle_50
    title: QuickSampling, Strebelle training image
    text: For each node, QS compares its data event with every position of the training image and draws one of the k best positions (n = 50, k = 2.4).
  - name: qs_stone_50
    title: QuickSampling, stone training image
    text: The same steps with a continuous variable; the mismatch is the mean squared difference over the neighbours (n = 50, k = 1.2).
  - name: ds_strebelle_50
    title: Direct Sampling, Strebelle training image
    text: DS scans the training image in random order and accepts the first candidate whose distance is under the threshold (n = 48, th = 0.12, f = 0.4).
  - name: ds_stone_50
    title: Direct Sampling, stone training image
    text: With a continuous variable, DS often scans the whole fraction f without finding a candidate under the threshold, and keeps the best one (n = 40, th = 0.08, f = 0.35).
  - name: snesim_strebelle_50
    title: SNESIM, Strebelle training image
    text: SNESIM first stores every pattern of the training image in a search tree, one per grid level; each node then only descends the tree with its data event and draws its category from the counts (template 7 × 7, 5 grid levels).
---

# Resources

## Presentation animations

These animations show how a sequential simulation works, one node at a time: visit the next node of the path, collect its data event, search the training image (or the SNESIM search tree), and paste the value. They replay real simulations: the algorithms ran through [pyDev]({{ site.baseurl }}/algorithms/PyDev.html), which logged every step, on a 50 × 50 grid.

The animations follow the light or dark appearance of your system. Each one can be downloaded as an animated SVG (for web pages) or as a video (MP4, 1920 × 1080, for slides; PowerPoint does not play animated SVGs). They can be regenerated, or made for other settings, with the scripts in [`example/python/pydev/animations`](https://github.com/GAIA-UNIL/G2S/tree/master/example/python/pydev/animations).

{% for anim in page.animations %}
{% assign light_svg = site.baseurl | append: "/assets/animations/" | append: anim.name | append: ".svg" %}
{% assign dark_svg = site.baseurl | append: "/assets/animations/" | append: anim.name | append: "_dark.svg" %}
{% assign light_mp4 = page.video_base | append: "/" | append: anim.name | append: ".mp4" %}
{% assign dark_mp4 = page.video_base | append: "/" | append: anim.name | append: "_dark.mp4" %}
<div class="animation-card" markdown="0">
  <h3>{{ anim.title }}</h3>
  <p>{{ anim.text }}</p>
  <picture>
    <source srcset="{{ dark_svg }}" media="(prefers-color-scheme: dark)">
    <img src="{{ light_svg }}" alt="Animation: {{ anim.title }}" loading="lazy" width="1920" height="1080">
  </picture>
  <div class="animation-downloads theme-light">
    <a class="download-button" href="{{ light_svg }}" download>Download SVG</a>
    <a class="download-button" href="{{ light_mp4 }}" download>Download video</a>
    <span class="other-theme">Dark background: <a href="{{ dark_svg }}" download>SVG</a> · <a href="{{ dark_mp4 }}" download>video</a></span>
  </div>
  <div class="animation-downloads theme-dark">
    <a class="download-button" href="{{ dark_svg }}" download>Download SVG</a>
    <a class="download-button" href="{{ dark_mp4 }}" download>Download video</a>
    <span class="other-theme">Light background: <a href="{{ light_svg }}" download>SVG</a> · <a href="{{ light_mp4 }}" download>video</a></span>
  </div>
</div>
{% endfor %}
