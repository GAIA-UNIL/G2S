/* G2S bridge page bootstrap - SPDX-License-Identifier: GPL-3.0-or-later */

import { startG2SLocalBridge } from "./g2s-bridge.js?v=20260723";

const status = document.querySelector("#status");
const progress = document.querySelector("#progress");
const progressLabel = document.querySelector("#progress-label");
const maximumThreads = document.querySelector("#max-threads");
const threadDetails = document.querySelector("#thread-details");
const hardwareThreads = Math.max(1, Number(navigator.hardwareConcurrency) || 1);
const threaded = window.crossOriginIsolated === true;
const availableThreads = threaded ? Math.min(8, hardwareThreads) : 1;
maximumThreads.max = String(availableThreads);
maximumThreads.value = String(Math.min(4, availableThreads));
maximumThreads.disabled = !threaded;
threadDetails.textContent = threaded
  ? `Multithreaded WebAssembly is available (${availableThreads} worker maximum).`
  : "Single-thread compatibility mode: serve this page with COOP/COEP headers to enable browser threads.";

const bridge = startG2SLocalBridge({
  maxThreads: Number(maximumThreads.value),
  workerUrl: new URL(
    threaded ? "./g2s-worker.js?threaded=1&v=20260723" : "./g2s-worker.js?v=20260723",
    import.meta.url,
  ),
  onStatus: (state, message) => {
    status.textContent = message;
    maximumThreads.disabled = !threaded || state === "running";
  },
  onProgress: ({ percent, message }) => {
    const value = Math.max(0, Math.min(100, Number(percent) || 0));
    progress.value = value;
    progressLabel.textContent = `${Math.round(value)}%${message ? ` — ${message}` : ""}`;
  },
});

function updateMaximumThreads() {
  const value = Math.max(1, Math.min(availableThreads, Math.floor(Number(maximumThreads.value) || 1)));
  maximumThreads.value = String(value);
  bridge.setMaxThreads(value);
}

maximumThreads.addEventListener("input", updateMaximumThreads);
maximumThreads.addEventListener("change", updateMaximumThreads);
