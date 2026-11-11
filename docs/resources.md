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
    svg_size: 1.2 MB
    mp4_size: 4.3 MB
    title: QuickSampling, Strebelle training image
    text: For each node, QS compares its data event with every position of the training image and draws one of the k best positions (n = 50, k = 2.4).
  - name: qs_stone_50
    svg_size: 1.3 MB
    mp4_size: 4.2 MB
    title: QuickSampling, stone training image
    text: The same steps with a continuous variable; the mismatch is the mean squared difference over the neighbours (n = 50, k = 1.2).
  - name: ds_strebelle_50
    svg_size: 1.2 MB
    mp4_size: 4.1 MB
    title: Direct Sampling, Strebelle training image
    text: DS scans the training image in random order and accepts the first candidate whose distance is under the threshold (n = 48, th = 0.12, f = 0.4).
  - name: ds_stone_50
    svg_size: 1.3 MB
    mp4_size: 6.2 MB
    title: Direct Sampling, stone training image
    text: With a continuous variable, DS often scans the whole fraction f without finding a candidate under the threshold, and keeps the best one (n = 40, th = 0.08, f = 0.35).
  - name: snesim_strebelle_50
    svg_size: 1.2 MB
    mp4_size: 2.5 MB
    title: SNESIM, Strebelle training image
    text: SNESIM first stores every pattern of the training image in a search tree, one per grid level; each node then only descends the tree with its data event and draws its category from the counts (template 7 × 7, 5 grid levels).
---

# Resources

## Presentation animations

These animations show how a sequential simulation works, one node at a time: visit the next node of the path, collect its data event, search the training image (or the SNESIM search tree), and paste the value. They replay real simulations: the algorithms ran through [pyDev]({{ site.baseurl }}/algorithms/PyDev.html), which logged every step, on a 50 × 50 grid.

The animations follow the light or dark appearance of your system; the switch under each one changes it. Each one can be downloaded as an animated SVG (for web pages) or as a video (MP4, 1920 × 1080, for slides; PowerPoint does not play animated SVGs). They can be regenerated, or made for other settings, with the scripts in [`example/python/pydev/animations`](https://github.com/GAIA-UNIL/G2S/tree/master/example/python/pydev/animations).

**Licence.** The animations are © 2026 Mathieu Gravey and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): you can use, share and adapt them, including in slides, courses and publications, as long as you give credit. Suggested credit: *© 2026 Mathieu Gravey · CC BY 4.0*. The author and licence are shown in the corner of each animation and written in the metadata of each SVG and video file.

{% for anim in page.animations %}
{% assign light_svg = site.baseurl | append: "/assets/animations/" | append: anim.name | append: ".svg" %}
{% assign dark_svg = site.baseurl | append: "/assets/animations/" | append: anim.name | append: "_dark.svg" %}
{% assign light_mp4 = page.video_base | append: "/" | append: anim.name | append: ".mp4" %}
{% assign dark_mp4 = page.video_base | append: "/" | append: anim.name | append: "_dark.mp4" %}
<div class="animation-card" markdown="0" data-light-svg="{{ light_svg }}" data-dark-svg="{{ dark_svg }}"
     data-light-mp4="{{ light_mp4 }}" data-dark-mp4="{{ dark_mp4 }}">
  <h3>{{ anim.title }}</h3>
  <p>{{ anim.text }}</p>
  <picture>
    <source srcset="{{ dark_svg }}" media="(prefers-color-scheme: dark)">
    <img src="{{ light_svg }}" alt="Animation: {{ anim.title }}" loading="lazy" width="1920" height="1080">
  </picture>
  <div class="animation-toolbar">
    <div class="animation-downloads">
      <a class="dl-btn dl-primary" data-kind="svg" href="{{ light_svg }}" download>
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5v8.5M4.5 6.5 8 10l3.5-3.5M2.5 13.5h11"/></svg>
        <span class="dl-label">SVG</span><span class="dl-meta">animated · {{ anim.svg_size }}</span>
      </a>
      <a class="dl-btn" data-kind="mp4" href="{{ light_mp4 }}" download>
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5v8.5M4.5 6.5 8 10l3.5-3.5M2.5 13.5h11"/></svg>
        <span class="dl-label">Video</span><span class="dl-meta">MP4 1080p · {{ anim.mp4_size }}</span>
      </a>
    </div>
    <div class="theme-switch" role="group" aria-label="Background of the animation" hidden>
      <button type="button" data-theme="light" aria-pressed="true">Light</button>
      <button type="button" data-theme="dark" aria-pressed="false">Dark</button>
    </div>
  </div>
</div>
{% endfor %}

<script src="{{ site.baseurl }}/assets/js/animations.js" defer></script>
