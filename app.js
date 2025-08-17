// app.js (ESM) — three r149 via CDN
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js';

console.log('[app] ESM mode');

const wrap = document.getElementById('canvas-wrap');
const fileInput = document.getElementById('file');
const depthScaleEl = document.getElementById('depthScale');
const strideEl = document.getElementById('stride');
const pauseEl = document.getElementById('pause');
const sampleBtn = document.getElementById('sample');

let renderer, scene, camera, controls, mesh, videoEl, canvasEl, ctx, videoTex, depthTex;
let worker, frameCount = 0, lastDepth = null;

const DEPTH_W = 256, DEPTH_H = 256;
const GEO_SEGS = 256;
const EMA = 0.6;
let depthScale = parseFloat(depthScaleEl.value || '0.25');
let stride = parseInt(strideEl.value || '1', 10);

initThree();
initWorker();
initVideoPipeline();
attachUI();
animate();

function initThree() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  wrap.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e11);

  camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 0.01, 100);
  camera.position.set(0, 0, 2.2);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.enableDamping = true;
  controls.minDistance = 0.6;
  controls.maxDistance = 6;
  controls.rotateSpeed = 0.6;
  controls.dampingFactor = 0.05;

  // start with a 16:9 plane; will resize when video loads
  const geo = new THREE.PlaneGeometry(16/9, 1, GEO_SEGS, GEO_SEGS);

  // WebGL1-safe single-channel float texture
  const depthData = new Float32Array(DEPTH_W * DEPTH_H).fill(0.5);
  const isGL2 = renderer.capabilities.isWebGL2;
  const depthFormat = isGL2 ? THREE.RedFormat : THREE.LuminanceFormat; // fallback for WebGL1
  depthTex = new THREE.DataTexture(depthData, DEPTH_W, DEPTH_H, depthFormat, THREE.FloatType);
  depthTex.needsUpdate = true;
  depthTex.flipY = false;

  // placeholder video texture (replaced when real video loads)
  const tmp = document.createElement('canvas'); tmp.width = 16; tmp.height = 9;
  const tctx = tmp.getContext('2d'); tctx.fillStyle = '#333'; tctx.fillRect(0,0,16,9);
  videoTex = new THREE.CanvasTexture(tmp);
  videoTex.flipY = false;

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uVideo: { value: videoTex },
      uDepth: { value: depthTex },
      uDepthScale: { value: depthScale }
    },
    vertexShader: `
      uniform sampler2D uDepth; uniform float uDepthScale; varying vec2 vUv;
      void main() {
        vUv = uv;
        float d = texture2D(uDepth, uv).r;
        vec3 displaced = position + normal * (d - 0.5) * uDepthScale;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
      }`,
    fragmentShader: `
      uniform sampler2D uVideo; varying vec2 vUv;
      void main() { gl_FragColor = texture2D(uVideo, vUv); }`
  });

  mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  window.addEventListener('resize', () => {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
}

function initWorker() {
  // Keep the worker "classic" for simplicity; path is relative to the HTML file.
  // When using ESM in the main thread, this still works:
  try {
    worker = new Worker('./depthWorker.js', { type: 'classic' });
  } catch (e) {
    console.error('[app] Failed to create Worker:', e);
    return;
  }

  worker.onmessage = (e) => {
    const { type } = e.data || {};
    if (type === 'depth') {
      let depth = e.data.depth;
      if (!(depth instanceof Float32Array)) depth = new Float32Array(depth); // normalize
      if (lastDepth && lastDepth.length === depth.length) {
        for (let i = 0; i < depth.length; i++) {
          depth[i] = EMA * depth[i] + (1 - EMA) * lastDepth[i];
        }
      }
      lastDepth = depth;
      depthTex.image.data.set(depth);
      depthTex.needsUpdate = true;
    } else {
      console.log('[app] worker msg:', e.data);
    }
  };

  worker.onerror = (e) => console.error('[app] worker error:', e.message || e);

  worker.postMessage({
    type: 'init',
    modelPath: './midas_small.onnx',
    inputSize: [DEPTH_W, DEPTH_H]
  });
}

function initVideoPipeline() {
  videoEl = document.createElement('video');
  videoEl.crossOrigin = 'anonymous';
  videoEl.muted = true; videoEl.playsInline = true; videoEl.loop = true;

  canvasEl = document.createElement('canvas');
  canvasEl.width = DEPTH_W; canvasEl.height = DEPTH_H;
  ctx = canvasEl.getContext('2d', { willReadFrequently: true });
}

function attachUI() {
  fileInput.addEventListener('change', async (e) => {
    const f = e.target.files && e.target.files[0]; if (!f) return;
    await loadVideo(URL.createObjectURL(f));
  });
  depthScaleEl.addEventListener('input', () => {
    depthScale = parseFloat(depthScaleEl.value || '0.25');
    mesh.material.uniforms.uDepthScale.value = depthScale;
  });
  strideEl.addEventListener('change', () => {
    stride = parseInt(strideEl.value || '1', 10);
    if (!stride || stride < 1) stride = 1;
  });
  sampleBtn.addEventListener('click', () => {
    let t = 0;
    const c = document.createElement('canvas'); c.width = 320; c.height = 180;
    const cctx = c.getContext('2d');
    (function tick(){
      cctx.fillStyle = `hsl(${(t*8)%360},70%,50%)`; cctx.fillRect(0,0,c.width,c.height);
      cctx.fillStyle = '#fff'; cctx.font = '18px system-ui'; cctx.fillText('Demo gradient (no video)', 16, 36);
      t++; videoTex.image = c; videoTex.needsUpdate = true; requestAnimationFrame(tick);
    })();
  });
}

function loadVideo(src) {
  return new Promise((resolve, reject) => {
    videoEl.src = src;
    videoEl.onloadedmetadata = async () => {
      try { await videoEl.play(); }
      catch (err) { console.warn('[app] Autoplay blocked; user gesture needed.', err); }
      const ar = (videoEl.videoWidth || 16) / (videoEl.videoHeight || 9);
      mesh.geometry.dispose();
      mesh.geometry = new THREE.PlaneGeometry(ar >= 1 ? ar : 1, ar >= 1 ? 1 : 1/ar, GEO_SEGS, GEO_SEGS);
      videoTex = new THREE.VideoTexture(videoEl);
      videoTex.flipY = false;
      mesh.material.uniforms.uVideo.value = videoTex;
      resolve();
    };
    videoEl.onerror = reject;
  });
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  renderer.render(scene, camera);

  if (videoEl && videoEl.readyState >= 2 && !(pauseEl && pauseEl.checked)) {
    if ((++frameCount % (stride || 1)) === 0) {
      ctx.drawImage(videoEl, 0, 0, DEPTH_W, DEPTH_H);
      const imageData = ctx.getImageData(0, 0, DEPTH_W, DEPTH_H);
      worker && worker.postMessage(
        { type: 'infer', width: DEPTH_W, height: DEPTH_H, data: imageData.data.buffer },
        [imageData.data.buffer]
      );
    }
  }
}

