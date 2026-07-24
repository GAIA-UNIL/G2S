/* Standalone QS deployment page - SPDX-License-Identifier: GPL-3.0-or-later */
import { G2S } from "./engine/g2s-api.js?v=deploy20260724g";
import { startG2SLocalBridge } from "./engine/g2s-bridge.js?v=deploy20260724g";

const $ = (selector) => document.querySelector(selector);
const logo = lottie.loadAnimation({
  container: $("#qs-logo"), renderer: "svg", loop: false, autoplay: false,
  path: "./assets/qs_logo.json", rendererSettings: { preserveAspectRatio: "xMidYMid meet" },
});

function setProgress(percent, message = "") {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  if (logo.totalFrames) logo.goToAndStop((value / 100) * (logo.totalFrames - 1), true);
  $("#progress-fill").style.width = `${value}%`;
  $("#percent").textContent = `${Math.round(value)}%`;
  $("#stage").textContent = message || (value >= 100 ? "Simulation complete" : "Ready for a simulation");
}
logo.addEventListener("DOMLoaded", () => setProgress(0, "Ready for a simulation"));

const threaded = window.crossOriginIsolated === true;
const hardware = Math.max(1, Number(navigator.hardwareConcurrency) || 1);
const available = threaded ? Math.min(8, hardware) : 1;
const threadInput = $("#demo-threads");
threadInput.max = String(available);
threadInput.value = String(Math.min(4, available));
$("#thread-note").textContent = threaded
  ? `Multithreaded Wasm available · ${available} worker maximum`
  : "Compatibility mode · serve with COOP/COEP headers to enable browser threads";

let bridge = null;
let lastBridgeError = "";
function createBridge() {
  return startG2SLocalBridge({
    maxThreads: Number(threadInput.value),
    workerUrl: new URL(threaded ? "./engine/g2s-worker.js?threaded=1&v=deploy20260724" : "./engine/g2s-worker.js?v=deploy20260724", import.meta.url),
    onStatus: (state, message) => {
      if (state === "error") {
        lastBridgeError = message;
        $("#bridge-status").textContent = `Browser bridge error: ${message}`;
      } else if (state === "waiting" && lastBridgeError) {
        $("#bridge-status").textContent = `Last browser bridge error: ${lastBridgeError}`;
      } else {
        $("#bridge-status").textContent = message;
      }
    },
    onProgress: ({ percent, message }) => setProgress(percent, message),
  });
}
if (navigator.locks?.request) {
  void navigator.locks.request("g2s-browser-bridge", { ifAvailable: true }, async (lock) => {
    if (!lock) {
      $("#bridge-status").textContent = "Another QS preview tab is active; close it before using Python or MATLAB.";
      return;
    }
    bridge = createBridge();
    await new Promise(() => {});
  });
} else {
  bridge = createBridge();
}

function selectedThreads() {
  const value = Math.max(1, Math.min(available, Math.floor(Number(threadInput.value) || 1)));
  threadInput.value = String(value);
  bridge?.setMaxThreads(value);
  return value;
}
threadInput.addEventListener("input", selectedThreads);
threadInput.addEventListener("change", selectedThreads);

const trainingImages = Object.freeze({
  stone: Object.freeze({
    name: "Stone",
    url: "./assets/Stone.png",
    variableType: "continuous",
  }),
  strebelle: Object.freeze({
    name: "Strebelle",
    url: "./assets/Strebelle.png",
    variableType: "categorical",
  }),
});

const trainingSelect = $("#training-image");
const trainingPreview = $("#training-preview");

function updateTrainingPreview() {
  const example = trainingImages[trainingSelect.value] || trainingImages.stone;
  trainingPreview.src = example.url;
  trainingPreview.alt = `${example.name} training image`;
  $("#training-description").textContent =
    `${example.name} · ${example.variableType} · the simulation will use the original image dimensions`;
}
trainingSelect.addEventListener("change", updateTrainingPreview);
updateTrainingPreview();

async function loadTrainingImage(example) {
  const image = new Image();
  image.decoding = "async";
  image.src = example.url;
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const values = new Float32Array(canvas.width * canvas.height);
  for (let index = 0; index < values.length; index += 1) {
    const value = rgba[index * 4] / 255;
    values[index] = example.variableType === "categorical" ? (value >= 0.5 ? 1 : 0) : value;
  }
  return { data: values, shape: [canvas.width, canvas.height] };
}

function randomSeed() {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return value[0] || 1;
}

function drawResult(data, shape) {
  const canvas = $("#result-canvas");
  const width = shape[0]; const height = shape[1];
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d"); const image = context.createImageData(width, height);
  let min = Infinity; let max = -Infinity;
  for (const value of data) { min = Math.min(min, value); max = Math.max(max, value); }
  const range = max > min ? max - min : 1;
  for (let i = 0; i < data.length; i += 1) { const shade = Math.round(255 * (data[i] - min) / range); image.data.set([shade, shade, shade, 255], i * 4); }
  context.putImageData(image, 0, 0);
}

let demoEngine = null;
let demoJob = null;
$("#run-demo").addEventListener("click", async () => {
  $("#run-demo").disabled = true; $("#cancel-demo").disabled = false; $("#demo-status").textContent = "Loading browser QS…"; setProgress(0, "Loading browser QS");
  try {
    const example = trainingImages[trainingSelect.value] || trainingImages.stone;
    const training = await loadTrainingImage(example);
    const seed = randomSeed();
    demoEngine ||= await G2S.create({ maxThreads: selectedThreads(), workerUrl: new URL(threaded ? "./engine/g2s-worker.js?threaded=1&v=deploy20260724" : "./engine/g2s-worker.js?v=deploy20260724", import.meta.url) });
    demoEngine.setMaxThreads(selectedThreads());
    demoJob = demoEngine.createJob();
    demoJob.loadArray("trainingImage", training.data, { shape: training.shape, variableTypes: [example.variableType] });
    demoJob.loadArray("destination", new Float32Array(training.data.length).fill(NaN), { shape: training.shape, variableTypes: [example.variableType] });
    demoJob.configure("qs", { candidates: 1.2, neighbors: [50], seed, mode: "vector", threads: selectedThreads() });
    demoJob.onProgress(({ percent, message }) => { setProgress(percent, message); $("#demo-status").textContent = message || `${Math.round(percent)}%`; });
    const result = await demoJob.run();
    const simulation = result.getArray("simulation");
    drawResult(simulation.data, simulation.shape);
    $("#result-note").textContent =
      `${example.name} · ${simulation.shape[0]}×${simulation.shape[1]} · seed ${seed} · ` +
      `${result.metadata.effective_threads || "1"} effective thread(s) · ${Math.round(result.durationMs)} ms`;
    $("#demo-status").textContent = "Browser simulation complete."; setProgress(100, "Simulation complete");
  } catch (error) {
    if (error?.name !== "AbortError") { $("#demo-status").textContent = `Demo error: ${error.message}`; setProgress(0, "Demo failed"); }
  } finally { demoJob = null; $("#run-demo").disabled = false; $("#cancel-demo").disabled = true; }
});
$("#cancel-demo").addEventListener("click", () => { demoJob?.cancel(); $("#demo-status").textContent = "Demo cancelled."; setProgress(0, "Ready for a simulation"); });
