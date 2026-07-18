/* G2S browser API - SPDX-License-Identifier: GPL-3.0-or-later */

const DEFAULT_WORKER_URL = new URL("./g2s-worker.js", import.meta.url);

function assertArrayDescriptor(data, descriptor) {
  if (!(data instanceof Float32Array)) {
    throw new TypeError("G2S input arrays must be Float32Array instances");
  }
  if (!descriptor || !Array.isArray(descriptor.shape) || descriptor.shape.length === 0) {
    throw new TypeError("A non-empty array shape is required");
  }
  if (!Array.isArray(descriptor.variableTypes) || descriptor.variableTypes.length === 0) {
    throw new TypeError("At least one variable type is required");
  }
  const elementCount = descriptor.shape.reduce((count, value) => {
    if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError("Array dimensions must be positive integers");
    return count * value;
  }, descriptor.variableTypes.length);
  if (elementCount !== data.length) {
    throw new RangeError(`Array contains ${data.length} values; metadata describes ${elementCount}`);
  }
  for (const type of descriptor.variableTypes) {
    if (type !== "continuous" && type !== "categorical" && type !== 0 && type !== 1) {
      throw new TypeError(`Unsupported variable type: ${type}`);
    }
  }
}

export class G2SResult {
  constructor(arrays, durationMs = 0, metadata = {}) {
    this.arrays = new Map(arrays.map((array) => [array.name, array]));
    this.durationMs = durationMs;
    this.metadata = Object.freeze({ ...metadata });
  }

  getArray(name) {
    const result = this.arrays.get(name);
    if (!result) throw new RangeError(`G2S result does not contain '${name}'`);
    return result;
  }
}

export class G2SJob {
  constructor(engine) {
    this.engine = engine;
    this.arrays = [];
    this.algorithm = null;
    this.options = {};
    this.progressHandler = null;
    this.activeRun = null;
  }

  loadArray(name, data, descriptor) {
    if (this.activeRun) throw new Error("Cannot change arrays while a job is running");
    assertArrayDescriptor(data, descriptor);
    this.arrays.push({
      name,
      data,
      shape: [...descriptor.shape],
      variableTypes: descriptor.variableTypes.map((type) => type === "categorical" || type === 1 ? 1 : 0),
    });
    return this;
  }

  configure(algorithm, options = {}) {
    if (this.activeRun) throw new Error("Cannot configure a running job");
    if (String(algorithm).toLowerCase() !== "qs" && String(algorithm).toLowerCase() !== "quicksampling") {
      throw new RangeError("The browser build currently supports only QS");
    }
    this.algorithm = "qs";
    this.options = { ...options, threads: 1 };
    return this;
  }

  onProgress(handler) {
    if (handler !== null && typeof handler !== "function") throw new TypeError("Progress handler must be a function");
    this.progressHandler = handler;
    return this;
  }

  async run() {
    if (!this.algorithm) throw new Error("Call configure('qs', options) before run()");
    if (!this.arrays.some((array) => array.name === "trainingImage")) throw new Error("At least one trainingImage is required");
    if (!this.arrays.some((array) => array.name === "destination")) throw new Error("A destination array is required");
    if (this.activeRun) throw new Error("This job is already running");

    const request = {
      algorithm: this.algorithm,
      options: this.options,
      arrays: this.arrays.map(({ name, data, shape, variableTypes }) => ({
        name, shape, variableTypes, buffer: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
      })),
    };
    this.activeRun = this.engine.run(request, this.progressHandler);
    try {
      return await this.activeRun;
    } finally {
      this.activeRun = null;
    }
  }

  cancel() {
    if (this.activeRun) this.engine.cancel();
  }
}

export class G2SEngine {
  constructor(workerUrl = DEFAULT_WORKER_URL) {
    this.workerUrl = workerUrl;
    this.worker = null;
    this.pending = null;
    this.workerReadyReject = null;
    this.generation = 0;
  }

  static async create(options = {}) {
    const engine = new G2SEngine(options.workerUrl || DEFAULT_WORKER_URL);
    await engine.ensureWorker();
    return engine;
  }

  createJob() { return new G2SJob(this); }

  ensureWorker() {
    if (this.worker) return Promise.resolve();
    this.worker = new Worker(this.workerUrl, { type: "module", name: "g2s-qs" });
    return new Promise((resolve, reject) => {
      this.workerReadyReject = reject;
      const ready = (event) => {
        if (event.data?.type === "ready") {
          this.worker.removeEventListener("message", ready);
          this.workerReadyReject = null;
          resolve();
        } else if (event.data?.type === "fatal") {
          this.worker.removeEventListener("message", ready);
          this.workerReadyReject = null;
          reject(new Error(event.data.message));
        }
      };
      this.worker.addEventListener("message", ready);
      this.worker.addEventListener("error", (event) => {
        this.workerReadyReject = null;
        reject(event.error || new Error(event.message));
      }, { once: true });
    });
  }

  async run(request, progressHandler) {
    const generation = this.generation;
    if (!this.worker) await this.ensureWorker();
    if (generation !== this.generation) throw new DOMException("G2S browser job cancelled", "AbortError");
    if (this.pending) throw new Error("The G2S engine already has an active job");
    const transfer = request.arrays.map((array) => array.buffer);
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject, progressHandler };
      this.worker.onmessage = (event) => {
        if (event.data?.type === "progress") {
          progressHandler?.(event.data.progress);
          return;
        }
        if (event.data?.type === "result") {
          const arrays = event.data.arrays.map((array) => ({
            ...array,
            data: array.encoding === "uint32" ? new Uint32Array(array.buffer) :
              array.encoding === "int32" ? new Int32Array(array.buffer) : new Float32Array(array.buffer),
          }));
          this.pending = null;
          resolve(new G2SResult(arrays, event.data.durationMs, event.data.metadata));
          return;
        }
        if (event.data?.type === "error") {
          this.pending = null;
          reject(new Error(event.data.message));
        }
      };
      this.worker.postMessage({ type: "run", request }, transfer);
    });
  }

  cancel() {
    this.generation += 1;
    if (this.worker) this.worker.terminate();
    this.worker = null;
    if (this.workerReadyReject) {
      this.workerReadyReject(new DOMException("G2S browser job cancelled", "AbortError"));
      this.workerReadyReject = null;
    }
    if (this.pending) {
      this.pending.reject(new DOMException("G2S browser job cancelled", "AbortError"));
      this.pending = null;
    }
  }

  dispose() { this.cancel(); }
}

export const G2S = Object.freeze({ create: (options) => G2SEngine.create(options) });
