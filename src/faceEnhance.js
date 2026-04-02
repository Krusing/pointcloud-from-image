import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// Nose tip — the very front of the nose (landmarks on the tip itself)
const NOSE_TIP_IDXS    = new Set([1, 4, 94]);

// Nose bridge — from mid-nose up toward the brow
const NOSE_BRIDGE_IDXS = new Set([6, 168, 195, 197]);

// Nose ala / wings / nostrils
const NOSE_ALA_IDXS    = new Set([2, 5, 19, 48, 98, 115, 129, 278, 344, 358]);

// Eye-socket contour landmarks
const EYE_IDXS = new Set([
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]);

// Lip contour landmarks
const MOUTH_IDXS = new Set([
  0, 11, 12, 13, 14, 15, 16, 17,
  37, 40, 61, 78, 80, 82, 87, 88, 91,
  267, 270, 291, 308, 310, 312, 317, 318, 321, 324,
  178, 181, 185, 191, 375, 402, 405, 409, 415,
]);

// sigma in normalised image-width units — nose_tip deliberately wide to cover the whole tip
const SIGMA = {
  nose_tip:    0.030,
  nose_bridge: 0.018,
  nose_ala:    0.012,
  eye:         0.012,
  mouth:       0.010,
  other:       0.012,
};

function getRegion(i) {
  if (NOSE_TIP_IDXS.has(i))    return 'nose_tip';
  if (NOSE_BRIDGE_IDXS.has(i)) return 'nose_bridge';
  if (NOSE_ALA_IDXS.has(i))    return 'nose_ala';
  if (EYE_IDXS.has(i))         return 'eye';
  if (MOUTH_IDXS.has(i))       return 'mouth';
  return 'other';
}

let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: 'IMAGE',
        numFaces: 1,
      });
    })();
  }
  return landmarkerPromise;
}

// Runs face landmark detection and computes per-region correction maps.
// Returns { faceFound, maps: {nose_tip, nose_bridge, nose_ala, eye, mouth, other}, faceBounds }
export async function computeFaceCorrections(imgEl, outW, outH) {
  const landmarker = await getLandmarker();
  const result = landmarker.detect(imgEl);
  if (!result.faceLandmarks?.length) return { faceFound: false };

  const pts = result.faceLandmarks[0];

  let zSum = 0;
  for (const p of pts) zSum += p.z;
  const zFaceMean = zSum / pts.length;

  const regions = ['nose_tip', 'nose_bridge', 'nose_ala', 'eye', 'mouth', 'other'];
  const acc = {}, wgt = {};
  for (const r of regions) {
    acc[r] = new Float32Array(outW * outH);
    wgt[r] = new Float32Array(outW * outH);
  }

  for (let i = 0; i < pts.length; i++) {
    const p      = pts[i];
    const region = getRegion(i);
    const corr   = -(p.z - zFaceMean); // positive = protrudes toward camera

    const sigma = SIGMA[region] * outW;
    const rr    = Math.ceil(sigma * 2.5);
    const sig2  = 2 * sigma * sigma;
    const cx    = p.x * outW, cy = p.y * outH;
    const x0 = Math.max(0, Math.floor(cx - rr)), x1 = Math.min(outW - 1, Math.ceil(cx + rr));
    const y0 = Math.max(0, Math.floor(cy - rr)), y1 = Math.min(outH - 1, Math.ceil(cy + rr));

    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const dx = px - cx, dy = py - cy;
        const w  = Math.exp(-(dx * dx + dy * dy) / sig2);
        const idx = py * outW + px;
        acc[region][idx] += w * corr;
        wgt[region][idx] += w;
      }
    }
  }

  const maps = {}, wmaps = {};
  for (const r of regions) {
    const m = new Float32Array(outW * outH);
    for (let i = 0; i < m.length; i++)
      if (wgt[r][i] > 0.01) m[i] = acc[r][i] / wgt[r][i];
    maps[r] = m;

    // Normalised weight map (0-1) — used for contrast-stretch blending
    let maxW = 0;
    for (let i = 0; i < wgt[r].length; i++) if (wgt[r][i] > maxW) maxW = wgt[r][i];
    const wm = new Float32Array(outW * outH);
    if (maxW > 0) for (let i = 0; i < wm.length; i++) wm[i] = wgt[r][i] / maxW;
    wmaps[r] = wm;
  }

  let fxMin = 1, fxMax = 0, fyMin = 1, fyMax = 0;
  for (const p of pts) {
    if (p.x < fxMin) fxMin = p.x; if (p.x > fxMax) fxMax = p.x;
    if (p.y < fyMin) fyMin = p.y; if (p.y > fyMax) fyMax = p.y;
  }

  // Raw per-landmark data for sculpt-style 3D correction (bypasses depth map entirely)
  const landmarks = [];
  for (let i = 0; i < pts.length; i++) {
    const region = getRegion(i);
    if (region === 'other') continue;
    landmarks.push({ region, x: pts[i].x, y: pts[i].y, corr: -(pts[i].z - zFaceMean) });
  }

  return { faceFound: true, maps, wmaps, faceBounds: { fxMin, fxMax, fyMin, fyMax }, landmarks };
}

// Applies pre-computed corrections to a base depth map.
// Uses additive corrections derived from MediaPipe z-coordinates — works even when the
// depth model gives a flat nose, because we use MediaPipe's own depth estimates.
// weight 0 = off, 1 = natural MediaPipe geometry, 2–5 = amplified.
export function applyFaceCorrections(baseDepth, outW, outH, corrections, weights) {
  const { maps, wmaps, faceBounds } = corrections;

  let dSum = 0, dN = 0;
  const fpx0 = Math.floor(faceBounds.fxMin * outW), fpx1 = Math.ceil(faceBounds.fxMax * outW);
  const fpy0 = Math.floor(faceBounds.fyMin * outH), fpy1 = Math.ceil(faceBounds.fyMax * outH);
  for (let py = fpy0; py <= fpy1; py++)
    for (let px = fpx0; px <= fpx1; px++) { dSum += baseDepth[py * outW + px]; dN++; }
  const faceMean = dN > 0 ? dSum / dN : 0.5;

  // Scale each map so its maximum absolute correction equals frac * faceMean,
  // then multiply by the user's weight slider.
  const scaleFor = (map, frac, w) => {
    let maxV = 0;
    for (let i = 0; i < map.length; i++) if (Math.abs(map[i]) > maxV) maxV = Math.abs(map[i]);
    return maxV > 0 ? (faceMean * frac) / maxV * w : 0;
  };

  const nts = scaleFor(maps.nose_tip,    0.20, weights.nose_tip    ?? 1);
  const nbs = scaleFor(maps.nose_bridge, 0.10, weights.nose_bridge ?? 1);
  const nas = scaleFor(maps.nose_ala,    0.07, weights.nose_ala    ?? 1);
  const es  = scaleFor(maps.eye,         0.07, weights.eye         ?? 1);
  const ms  = scaleFor(maps.mouth,       0.05, weights.mouth       ?? 1);
  const os  = scaleFor(maps.other,       0.04, 1);

  // Multiply each correction by wmaps (the raw Gaussian bell shape).
  // maps[r][i] = acc/wgt = constant across the region (flat hat without wmaps).
  // maps[r][i] * wmaps[r][i] = bell-shaped bump, max at landmark centre, smooth falloff.
  const out = new Float32Array(baseDepth);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, Math.min(1,
      baseDepth[i]
        + maps.nose_tip[i]    * nts * wmaps.nose_tip[i]
        + maps.nose_bridge[i] * nbs * wmaps.nose_bridge[i]
        + maps.nose_ala[i]    * nas * wmaps.nose_ala[i]
        + maps.eye[i]         * es  * wmaps.eye[i]
        + maps.mouth[i]       * ms  * wmaps.mouth[i]
        + maps.other[i]       * os
    ));
  }
  return out;
}
