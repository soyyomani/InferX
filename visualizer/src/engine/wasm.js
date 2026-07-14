// Loads the C++ WASM module via script tag (Vite can't import from /public).
// All tensor logic runs in actual compiled C++ code.

let module = null;
let loadingPromise = null;

export async function initWasm() {
  if (module) return module;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    // Only add script if not already present
    if (!window.createInferXModule) {
      await new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "/inferx_wasm.js";
        script.onload = resolve;
        script.onerror = () => reject(new Error("Failed to load inferx_wasm.js"));
        document.head.appendChild(script);
      });
    }

    // Wait for the factory function to appear
    let retries = 0;
    while (!window.createInferXModule && retries < 50) {
      await new Promise(r => setTimeout(r, 100));
      retries++;
    }

    if (!window.createInferXModule) {
      throw new Error("createInferXModule not found after loading script");
    }

    module = await window.createInferXModule();
    return module;
  })();

  return loadingPromise;
}

export function isReady() {
  return module !== null;
}

// Convert VectorTraceStep to JS array
function traceVecToArray(vec) {
  const arr = [];
  for (let i = 0; i < vec.size(); i++) {
    const step = vec.get(i);
    const internal = [];
    const internalVec = step.internal;
    for (let j = 0; j < internalVec.size(); j++) {
      internal.push(internalVec.get(j));
    }
    arr.push({
      component: step.component,
      title: step.title,
      detail: step.detail,
      internal,
    });
  }
  vec.delete();
  return arr;
}

function toInt64Vec(arr) {
  const vec = new module.VectorInt64();
  for (const v of arr) vec.push_back(v);
  return vec;
}

// Public API

export function traceFullCreate(dtypeName, dims, fillMode) {
  if (!module) throw new Error("WASM not loaded");
  const vec = toInt64Vec(dims);
  const result = module.traceFullCreate(dtypeName, vec, fillMode);
  vec.delete();
  return traceVecToArray(result);
}

export function traceAccess(dims, indices) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const iv = toInt64Vec(indices);
  const result = module.traceAccess(dv, iv);
  dv.delete(); iv.delete();
  return traceVecToArray(result);
}

export function traceReshape(oldDims, newDims) {
  if (!module) throw new Error("WASM not loaded");
  const ov = toInt64Vec(oldDims);
  const nv = toInt64Vec(newDims);
  const result = module.traceReshape(ov, nv);
  ov.delete(); nv.delete();
  return traceVecToArray(result);
}

export function traceSlice(dims, dim, start, end) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const result = module.traceSlice(dv, dim, start, end);
  dv.delete();
  return traceVecToArray(result);
}

export function traceTranspose(dims, dim0, dim1) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const result = module.traceTranspose(dv, dim0, dim1);
  dv.delete();
  return traceVecToArray(result);
}

export function traceBroadcast(dimsA, dimsB) {
  if (!module) throw new Error("WASM not loaded");
  const va = toInt64Vec(dimsA);
  const vb = toInt64Vec(dimsB);
  const result = module.traceBroadcast(va, vb);
  va.delete(); vb.delete();
  return traceVecToArray(result);
}

export function traceIterator(dims, transposed) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const result = module.traceIterator(dv, transposed);
  dv.delete();
  return traceVecToArray(result);
}

export function traceContiguous(dims) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const result = module.traceContiguous(dv);
  dv.delete();
  return traceVecToArray(result);
}

export function traceClone(dims) {
  if (!module) throw new Error("WASM not loaded");
  const dv = toInt64Vec(dims);
  const result = module.traceClone(dv);
  dv.delete();
  return traceVecToArray(result);
}
