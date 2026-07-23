/* G2S QS Web Worker - SPDX-License-Identifier: GPL-3.0-or-later */

const workerParameters = new URL(globalThis.location.href).searchParams;
const threaded = workerParameters.get("threaded") === "1";
const threadPoolSize = threaded ? 8 : 1;
const moduleUrl = threaded ? "../dist/g2s-qs-pthreads.mjs" : "../dist/g2s-qs.mjs";

let modulePromise;

function getModule() {
  if (!modulePromise) {
    modulePromise = import(moduleUrl).then(({ default: createG2SModule }) =>
      createG2SModule({
        noInitialRun: true,
        printErr: (message) => postMessage({ type: "diagnostic", message }),
      }));
  }
  return modulePromise;
}

async function run(request) {
  const module = await getModule();
  const started = performance.now();
  const progress = (percent, message = "") => postMessage({
    type: "progress",
    progress: { percent, message },
  });

  const nativeRequest = {
    algorithm: request.algorithm,
    options: request.options,
    arrays: request.arrays.map((array) => ({
      name: array.name,
      shape: array.shape,
      variableTypes: array.variableTypes,
      data: new Float32Array(array.buffer),
    })),
  };
  const nativeResult = module.runQs(nativeRequest, progress);
  if (nativeResult.error) throw new Error(nativeResult.error);
  const metadata = {
    ...(nativeResult.metadata || {}),
    requested_threads: String(request.options.requestedThreads ?? 1),
    effective_threads: String(request.options.threads ?? 1),
    browser_max_threads: String(request.options.maximumThreads ?? 1),
  };
  if (request.options.threadRequestClamped) {
    metadata.thread_warning =
      `Requested ${request.options.requestedThreads} threads; browser policy limited the job to ${request.options.threads}.`;
  }
  const arrays = nativeResult.arrays.map((array) => {
    const typed = array.encoding === "uint32" ? new Uint32Array(array.data) :
      array.encoding === "int32" ? new Int32Array(array.data) : new Float32Array(array.data);
    return {
      name: array.name,
      shape: Array.from(array.shape),
      variableTypes: Array.from(array.variableTypes),
      encoding: array.encoding || "float32",
      buffer: typed.buffer,
    };
  });
  postMessage({
    type: "result",
    arrays,
    durationMs: nativeResult.durationMs || performance.now() - started,
    metadata,
  }, arrays.map((array) => array.buffer));
}

self.onmessage = async (event) => {
  if (event.data?.type !== "run") return;
  try {
    await run(event.data.request);
  } catch (error) {
    postMessage({ type: "error", message: error?.message || String(error) });
  }
};

getModule().then(
  () => postMessage({ type: "ready", capabilities: { threaded, threadPoolSize } }),
  (error) => postMessage({ type: "fatal", message: error?.message || String(error) }),
);
