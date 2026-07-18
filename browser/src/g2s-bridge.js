/* G2S on-demand localhost bridge - SPDX-License-Identifier: GPL-3.0-or-later */

import { G2S } from "./g2s-api.js";

const ARRAY_NAMES = Object.freeze({
  "-ti": "trainingImage", "-di": "destination", "-ki": "kernel",
  "-sp": "simulationPath", "-ii": "trainingImageIndex", "-ni": "neighborCountMap",
  "-kii": "kernelIndexMap", "-kvi": "candidateCountMap", "-rmi": "rotationMap",
  "-smi": "scaleMap",
});

const first = (parameters, name, fallback) => parameters[name]?.[0] ?? fallback;
const numbers = (parameters, name) => (parameters[name] || []).map(Number);
const flag = (parameters, name) => Object.prototype.hasOwnProperty.call(parameters, name);

function typedOptions(parameters) {
  return {
    candidates: Number(first(parameters, "-k", NaN)),
    maximumExplorationRatio: Number(first(parameters, "-f", first(parameters, "-mer", NaN))),
    neighbors: numbers(parameters, "-n"),
    seed: Number(first(parameters, "-s", 0)),
    mode: flag(parameters, "-fs") ? "full" : "vector",
    forceSimulation: flag(parameters, "--forceSimulation"),
    circularTrainingImage: flag(parameters, "-cti"),
    circularSimulation: flag(parameters, "-csim"),
    noVerbatim: flag(parameters, "-nV"),
    fullStationary: flag(parameters, "-far") || flag(parameters, "-fastAndRisky"),
    pathOptimization: flag(parameters, "-wPO"),
    maximumNeighborhood: flag(parameters, "-maxNK"),
    distance: flag(parameters, "-wd") ? "kernel" : flag(parameters, "-md") ? "manhattan" : "euclidean",
    kernelSize: Number(first(parameters, "-ks", NaN)),
    alpha: Number(first(parameters, "-alpha", NaN)),
    threads: 1,
  };
}

function requestInit(init = {}) {
  try { return { ...init, mode: "cors", targetAddressSpace: "loopback" }; }
  catch { return { ...init, mode: "cors" }; }
}

export class G2SLocalBridge {
  constructor({ endpoint = "http://127.0.0.1:8129", pollIntervalMs = 1000, workerUrl, onStatus } = {}) {
    this.endpoint = endpoint.replace(/\/$/, "");
    this.pollIntervalMs = pollIntervalMs;
    this.workerUrl = workerUrl;
    this.onStatus = typeof onStatus === "function" ? onStatus : () => {};
    this.stopped = true;
    this.activeJob = null;
    this.engine = null;
  }

  start() {
    if (!this.stopped) return;
    this.stopped = false;
    this.onStatus("waiting", "Waiting for a local QS command.");
    void this.poll();
  }

  stop() {
    this.stopped = true;
    this.activeJob?.cancel();
    this.engine?.dispose();
    this.engine = null;
  }

  async fetch(path, nonce, init = {}) {
    const headers = new Headers(init.headers || {});
    if (nonce) {
      headers.set("X-G2S-Nonce", nonce);
      headers.set("X-G2S-Protocol-Version", String(this.activeSession?.protocolVersion ?? 1));
      headers.set("X-G2S-Session-Id", this.activeSession?.sessionId || "");
    }
    return fetch(`${this.endpoint}${path}`, requestInit({ ...init, headers, cache: "no-store" }));
  }

  async poll() {
    while (!this.stopped) {
      try {
        const response = await this.fetch("/v1/session");
        if (response.ok) await this.handleSession(await response.json());
        else this.onStatus("waiting", `Local bridge returned HTTP ${response.status}.`);
      } catch {
        // Connection refusal is normal while no Python/MATLAB command is active.
        this.onStatus("waiting", "Ready; waiting for a local QS command.");
      }
      await new Promise((resolve) => setTimeout(resolve, this.pollIntervalMs));
    }
  }

  async handleSession(session) {
    if (this.activeJob) return;
    if (session.protocolVersion !== 1 || typeof session.sessionId !== "string" || typeof session.nonce !== "string") {
      throw new Error("Unsupported or malformed G2S browser session");
    }
    const nonce = session.nonce;
    this.activeSession = session;
    this.onStatus("running", `Running QS session ${session.sessionId}.`);
    let controlTimer;
    let cancelRequested = false;
    try {
      // Start heartbeats before loading/compiling Wasm; a cold module load must
      // not look like a disconnected page to the synchronous interface.
      controlTimer = setInterval(async () => {
        try {
          const response = await this.fetch("/v1/control", nonce);
          if (response.ok && (await response.json()).cancel) {
            cancelRequested = true;
            this.activeJob?.cancel();
          }
        } catch { this.activeJob?.cancel(); }
      }, 1000);

      const manifestResponse = await this.fetch("/v1/job", nonce);
      if (!manifestResponse.ok) throw new Error(`Unable to retrieve G2S job (${manifestResponse.status})`);
      const manifest = await manifestResponse.json();
      const engine = this.engine || await G2S.create({ workerUrl: this.workerUrl });
      this.engine = engine;
      if (cancelRequested) throw new DOMException("G2S browser job cancelled", "AbortError");
      const job = engine.createJob();
      this.activeJob = job;

      for (const descriptor of manifest.arrays) {
        const response = await this.fetch(`/v1/arrays/${encodeURIComponent(descriptor.id)}`, nonce);
        if (!response.ok) throw new Error(`Unable to retrieve G2S array '${descriptor.id}'`);
        const data = new Float32Array(await response.arrayBuffer());
        job.loadArray(ARRAY_NAMES[descriptor.parameter] || descriptor.parameter, data, {
          shape: descriptor.dimensions,
          variableTypes: descriptor.variableTypes,
        });
      }
      job.configure("qs", typedOptions(manifest.parameters));
      let lastProgressAt = 0;
      job.onProgress((progress) => {
        const now = performance.now();
        if (now - lastProgressAt < 200 && progress.percent < 100) return;
        lastProgressAt = now;
        void this.fetch("/v1/progress", nonce, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(progress),
        });
      });
      const result = await job.run();
      for (const [name, descriptor] of result.arrays) {
        const types = descriptor.variableTypes.join(",");
        const dimensions = descriptor.shape.join(",");
        const response = await this.fetch(`/v1/results/${encodeURIComponent(name)}`, nonce, {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
            "X-G2S-Dimensions": dimensions,
            "X-G2S-Variable-Types": types,
            "X-G2S-Encoding": descriptor.encoding,
          },
          body: descriptor.data,
        });
        if (!response.ok) throw new Error(`Unable to upload G2S result '${name}'`);
      }
      await this.fetch("/v1/complete", nonce, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ durationMs: result.durationMs, metadata: result.metadata }),
      });
    } catch (error) {
      try {
        await this.fetch("/v1/error", nonce, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: error?.message || String(error) }),
        });
      } catch { /* The interface may already have timed out. */ }
    } finally {
      clearInterval(controlTimer);
      this.activeJob = null;
      this.activeSession = null;
      this.onStatus("waiting", "Waiting for a local QS command.");
    }
  }
}

export function startG2SLocalBridge(options) {
  const bridge = new G2SLocalBridge(options);
  bridge.start();
  return bridge;
}
