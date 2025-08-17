// depthWorker.js — ONNX Runtime Web in a Worker to keep UI smooth
console.log('[worker] started');

// Load ORT (UMD build)
self.importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js');

// ✅ Tell ORT where the WASM files live (CDN or your own folder)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
ort.env.wasm.numThreads = Math.max(1, (self.navigator?.hardwareConcurrency || 4) - 1);
ort.env.wasm.simd = true;

let session = null;
let IN_W = 256, IN_H = 256;

self.onmessage = async (e) => {
  const { type } = e.data || {};

  if (type === 'init') {
    const { modelPath, inputSize } = e.data;
    IN_W = inputSize[0]; IN_H = inputSize[1];

    console.log('[worker] loading model', modelPath);
    try {
      // ✅ Ask ONLY for WASM (don’t request webgpu/webgl)
      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('[worker] model loaded');
      postMessage({ type: 'ready' });
    } catch (err) {
      console.error('[worker] init failed', err);
      postMessage({ type: 'error', msg: `init failed: ${err?.message || err}` });
    }
    return;
  }

  if (type === 'infer') {
    if (!session) return;

    try {
      const rgba = new Uint8ClampedArray(e.data.data); // RGBA interleaved
      const chw = new Float32Array(1 * 3 * IN_H * IN_W);
      for (let y = 0; y < IN_H; y++) {
        for (let x = 0; x < IN_W; x++) {
          const p = (y * IN_W + x) * 4;
          const r = rgba[p] / 255, g = rgba[p+1] / 255, b = rgba[p+2] / 255;
          // NCHW layout
          chw[(0 * IN_H + y) * IN_W + x] = r;
          chw[(1 * IN_H + y) * IN_W + x] = g;
          chw[(2 * IN_H + y) * IN_W + x] = b;
        }
      }
      const input = new ort.Tensor('float32', chw, [1,3,IN_H,IN_W]);

      const feeds = {};
      feeds[session.inputNames[0]] = input;

      const out = await session.run(feeds);
      const outName = session.outputNames[0];
      const raw = out[outName].data; // Float32Array
      const N = IN_W * IN_H;

      let min = Infinity, max = -Infinity;
      const len = Math.min(raw.length, N);
      for (let i = 0; i < len; i++) { const v = raw[i]; if (v < min) min = v; if (v > max) max = v; }
      const eps = (max - min) || 1;

      const depth = new Float32Array(N);
      for (let i = 0; i < N; i++) depth[i] = (raw[i] - min) / eps;

      postMessage({ type: 'depth', width: IN_W, height: IN_H, depth }, [depth.buffer]);
    } catch (err) {
      console.error('[worker] infer failed', err);
      postMessage({ type: 'error', msg: `infer failed: ${err?.message || err}` });
    }
  }
};

