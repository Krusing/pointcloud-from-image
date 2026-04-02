import { useEffect, useRef, useState, useCallback } from "react";

// ─── Dual-handle range slider ──────────────────────────────────────────────
function RangeClip({ minVal, maxVal, onChange }) {
  const trackRef = useRef(null);
  const dragging = useRef(null);

  const getPct = useCallback((e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }, []);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      const p = getPct(e);
      if (dragging.current === "min") onChange(Math.min(p, maxVal - 0.02), maxVal);
      else                            onChange(minVal, Math.max(p, minVal + 0.02));
    };
    const onUp = () => { dragging.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [minVal, maxVal, onChange, getPct]);

  const thumb = (side, val) => (
    <div
      key={side}
      onMouseDown={e => { dragging.current = side; e.preventDefault(); }}
      onTouchStart={e => { dragging.current = side; e.preventDefault(); }}
      style={{
        position: "absolute", left: `${val * 100}%`, transform: "translateX(-50%)",
        width: 18, height: 18, borderRadius: "50%",
        background: "#fff", boxShadow: "0 1px 6px rgba(0,0,0,.6)",
        cursor: "grab", zIndex: 2, touchAction: "none",
      }}
    />
  );

  return (
    <div ref={trackRef} style={{ position: "relative", height: 18, display: "flex", alignItems: "center" }}>
      <div style={{ position: "absolute", left: 0, right: 0, height: 4, background: "rgba(255,255,255,.1)", borderRadius: 2 }} />
      <div style={{
        position: "absolute",
        left: `${minVal * 100}%`, right: `${(1 - maxVal) * 100}%`,
        height: 4, background: "#00d4ff", borderRadius: 2,
      }} />
      {thumb("min", minVal)}
      {thumb("max", maxVal)}
    </div>
  );
}

// ─── 3D view ───────────────────────────────────────────────────────────────
export default function Depth3DView({ depthData, depthW, depthH, imageCanvas, onClose, baseDepth = null, faceCorrections = null }) {
  const canvasRef = useRef(null);

  const [depthScale,      setDepthScale]      = useState(0.35);
  const [splitThreshold,  setSplitThreshold]  = useState(0.15);
  const [clipMin,         setClipMin]         = useState(0);
  const [clipMax,         setClipMax]         = useState(1);
  const [showControls,    setShowControls]    = useState(true);
  const [pointMode,       setPointMode]       = useState(false);
  const [pointSize,       setPointSize]       = useState(2);
  const [sculptMode,      setSculptMode]      = useState(false);
  const [brushSize,       setBrushSize]       = useState(60);
  const [brushStrength,   setBrushStrength]   = useState(3);
  const [eraseMode,       setEraseMode]       = useState(false);
  const [faceMode,        setFaceMode]        = useState(false);

  const [faceSliders, setFaceSliders] = useState({ nose_tip: 0, nose_bridge: 0, nose_ala: 0, eye: 0, mouth: 0 });
  const faceSlidersRef = useRef({ nose_tip: 0, nose_bridge: 0, nose_ala: 0, eye: 0, mouth: 0 });
  const depthsRef    = useRef(null);
  const depthBufRef  = useRef(null);
  const glRef        = useRef(null);

  const depthScaleRef     = useRef(0.35);
  const splitThresholdRef = useRef(0.15);
  const prevSplitRef      = useRef(-1);   // force initial index build
  const clipMinRef        = useRef(0);
  const clipMaxRef        = useRef(1);
  const pointModeRef      = useRef(false);
  const pointSizeRef      = useRef(2);
  const sculptModeRef     = useRef(false);
  const brushSizeRef      = useRef(60);
  const brushStrengthRef  = useRef(3);
  const eraseModeRef      = useRef(false);
  const currentMVPRef     = useRef(null);

  const handleClipChange = useCallback((lo, hi) => {
    setClipMin(lo); setClipMax(hi);
    clipMinRef.current = lo; clipMaxRef.current = hi;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !depthData || !imageCanvas || imageCanvas.width === 0) return;

    const gl = canvas.getContext("webgl");
    if (!gl) return;
    gl.getExtension("OES_element_index_uint");

    let canvasRect = canvas.getBoundingClientRect();
    let brushX = 0, brushY = 0;
    function resize() {
      canvas.width  = canvas.clientWidth  * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      gl.viewport(0, 0, canvas.width, canvas.height);
      canvasRect = canvas.getBoundingClientRect();
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // ── Shaders ──────────────────────────────────────────────────────────
    function mkShader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, mkShader(gl.VERTEX_SHADER, `
      attribute vec2 aXY;
      attribute float aDepth;
      attribute vec2 aUV;
      attribute float aErased;
      uniform mat4 uMVP;
      uniform float uDepthScale;
      uniform float uPointSize;
      uniform vec2  uBrushNDC;
      uniform float uBrushRadiusNDC;
      uniform float uSculptActive;
      varying vec2 vUV;
      varying float vDepth;
      varying float vErased;
      varying float vHighlight;
      void main() {
        vec4 pos = uMVP * vec4(aXY.x, aXY.y, aDepth * uDepthScale, 1.0);
        gl_Position = pos;
        gl_PointSize = uPointSize;
        vUV = aUV;
        vDepth = aDepth;
        vErased = aErased;
        if (uSculptActive > 0.5) {
          vec2 ndc = pos.xy / pos.w;
          vHighlight = length(ndc - uBrushNDC) < uBrushRadiusNDC ? 1.0 : 0.0;
        } else {
          vHighlight = 0.0;
        }
      }
    `));
    gl.attachShader(prog, mkShader(gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform sampler2D uTex;
      uniform float uClipMin;
      uniform float uClipMax;
      uniform float uEraseMode;
      varying vec2 vUV;
      varying float vDepth;
      varying float vErased;
      varying float vHighlight;
      void main() {
        if (vErased > 0.5) discard;
        if (vDepth < uClipMin || vDepth > uClipMax) discard;
        vec4 color = texture2D(uTex, vUV);
        if (vHighlight > 0.5) {
          vec4 hlColor = uEraseMode > 0.5 ? vec4(1.0, 0.2, 0.2, 1.0) : vec4(0.0, 0.8, 1.0, 1.0);
          color = mix(color, hlColor, 0.4);
        }
        gl_FragColor = color;
      }
    `));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    // ── Mesh geometry ────────────────────────────────────────────────────
    const GRID  = Math.min(256, depthW, depthH);
    const stepX = (depthW - 1) / (GRID - 1);
    const stepY = (depthH - 1) / (GRID - 1);
    const aspect = depthW / depthH;

    const xy     = new Float32Array(GRID * GRID * 2);
    const depths = new Float32Array(GRID * GRID);
    const uvs    = new Float32Array(GRID * GRID * 2);

    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const idx  = j * GRID + i;
        const srcX = Math.round(i * stepX);
        const srcY = Math.round(j * stepY);
        xy[idx * 2]     = (i / (GRID - 1) - 0.5) * aspect;
        xy[idx * 2 + 1] = -(j / (GRID - 1) - 0.5);
        depths[idx]     = depthData[srcY * depthW + srcX];
        uvs[idx * 2]     = i / (GRID - 1);
        uvs[idx * 2 + 1] = j / (GRID - 1);  // no flip — canvas Y=0 → top → V=0
      }
    }

    // ── Buffers ───────────────────────────────────────────────────────────
    const xyBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, xyBuf);
    gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);

    const depthBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, depthBuf);
    gl.bufferData(gl.ARRAY_BUFFER, depths, gl.DYNAMIC_DRAW);
    depthsRef.current   = depths;
    depthBufRef.current = depthBuf;
    glRef.current       = gl;

    const uvBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

    const erased  = new Float32Array(GRID * GRID);
    const erasedBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, erasedBuf);
    gl.bufferData(gl.ARRAY_BUFFER, erased, gl.DYNAMIC_DRAW);

    const maxIndices = (GRID - 1) * (GRID - 1) * 6;
    const indices    = new Uint32Array(maxIndices);
    const idxBuf     = gl.createBuffer();
    let   indexCount = 0;

    function rebuildIndices(threshold) {
      let k = 0;
      for (let j = 0; j < GRID - 1; j++) {
        for (let i = 0; i < GRID - 1; i++) {
          const tl = j * GRID + i, tr = tl + 1, bl = tl + GRID, br = bl + 1;
          if (erased[tl] || erased[tr] || erased[bl] || erased[br]) continue;
          const d0 = depths[tl], d1 = depths[tr], d2 = depths[bl], d3 = depths[br];
          const maxDiff = Math.max(
            Math.abs(d0-d1), Math.abs(d0-d2), Math.abs(d0-d3),
            Math.abs(d1-d2), Math.abs(d1-d3), Math.abs(d2-d3)
          );
          if (maxDiff > threshold) continue;
          indices[k++] = tl; indices[k++] = bl; indices[k++] = tr;
          indices[k++] = tr; indices[k++] = bl; indices[k++] = br;
        }
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices.subarray(0, k), gl.DYNAMIC_DRAW);
      indexCount = k;
      prevSplitRef.current = threshold;
    }

    // ── Texture ───────────────────────────────────────────────────────────
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── Locations ─────────────────────────────────────────────────────────
    const aXYLoc    = gl.getAttribLocation(prog, "aXY");
    const aDepthLoc = gl.getAttribLocation(prog, "aDepth");
    const aUVLoc    = gl.getAttribLocation(prog, "aUV");
    const uMVPLoc   = gl.getUniformLocation(prog, "uMVP");
    const uScaleLoc = gl.getUniformLocation(prog, "uDepthScale");
    const uMinLoc        = gl.getUniformLocation(prog, "uClipMin");
    const uMaxLoc        = gl.getUniformLocation(prog, "uClipMax");
    const uPSizeLoc      = gl.getUniformLocation(prog, "uPointSize");
    const aErasedLoc     = gl.getAttribLocation(prog,  "aErased");
    const uBrushNDCLoc   = gl.getUniformLocation(prog, "uBrushNDC");
    const uBrushRadLoc   = gl.getUniformLocation(prog, "uBrushRadiusNDC");
    const uSculptActLoc  = gl.getUniformLocation(prog, "uSculptActive");
    const uEraseModeULoc = gl.getUniformLocation(prog, "uEraseMode");

    // ── Sculpt helpers ────────────────────────────────────────────────────
    function projectToScreen(mvp, x, y, z) {
      const cx = mvp[0]*x + mvp[4]*y + mvp[8]*z  + mvp[12];
      const cy = mvp[1]*x + mvp[5]*y + mvp[9]*z  + mvp[13];
      const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
      const r  = canvas.getBoundingClientRect();
      return [(cx/cw + 1) * 0.5 * r.width, (1 - cy/cw) * 0.5 * r.height];
    }
    function applyBrush(clientX, clientY, delta) {
      const mvp = currentMVPRef.current;
      if (!mvp) return;
      const br    = brushSizeRef.current;
      const br2   = br * br;
      const sig2  = (br / 2.5) * (br / 2.5);
      const mx    = clientX - canvasRect.left;
      const my    = clientY - canvasRect.top;
      const erase = eraseModeRef.current;
      let changedDepth = false, changedErased = false;
      for (let idx = 0; idx < GRID * GRID; idx++) {
        const vx = xy[idx * 2], vy = xy[idx * 2 + 1];
        const [sx, sy] = projectToScreen(mvp, vx, vy, depths[idx] * depthScaleRef.current);
        const d2 = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        if (d2 > br2) continue;
        if (erase) {
          erased[idx] = 1;
          changedErased = true;
        } else {
          const falloff = Math.exp(-d2 / (2 * sig2));
          depths[idx] = Math.max(0, Math.min(1, depths[idx] + delta * brushStrengthRef.current * falloff));
          changedDepth = true;
        }
      }
      if (changedDepth) {
        gl.bindBuffer(gl.ARRAY_BUFFER, depthBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, depths);
      }
      if (changedErased) {
        gl.bindBuffer(gl.ARRAY_BUFFER, erasedBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, erased);
      }
      if (changedDepth || changedErased) prevSplitRef.current = -1;
    }

    // ── Matrix math (column-major) ────────────────────────────────────────
    function norm3(v) { const l=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }
    function cross3(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
    function dot3(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

    function perspective(fov, asp, near, far) {
      const f = 1 / Math.tan(fov / 2);
      return new Float32Array([
        f/asp, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (near+far)/(near-far), -1,
        0, 0, 2*near*far/(near-far), 0,
      ]);
    }
    function lookAt(eye, center, up) {
      const z = norm3([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
      const x = norm3(cross3(up, z));
      const y = cross3(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot3(x,eye), -dot3(y,eye), -dot3(z,eye), 1,
      ]);
    }
    function matMul(a, b) {
      const m = new Float32Array(16);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
          let s = 0;
          for (let kk = 0; kk < 4; kk++) s += a[kk*4+i] * b[j*4+kk];
          m[j*4+i] = s;
        }
      return m;
    }

    // ── Free-look camera ──────────────────────────────────────────────────
    // Camera rotates around its own axes. Scroll moves forward/back.
    const camPos = [0, 0.2, 2.0];
    let yaw = 0, pitch = -0.1;   // yaw: horizontal, pitch: vertical (clamped ±80°)
    let leftDown = false, rightDown = false;
    let lastX = 0, lastY = 0, lastTouchDist = 0, lastTouchMidX = 0, lastTouchMidY = 0;
    const keys = new Set();

    function getLookDir() {
      return [
        Math.sin(yaw) * Math.cos(pitch),
        Math.sin(pitch),
        -Math.cos(yaw) * Math.cos(pitch),
      ];
    }
    function getCamRight() {
      return norm3(cross3(getLookDir(), [0, 1, 0]));
    }
    function getCamUp() {
      return cross3(getCamRight(), getLookDir());
    }

    const onMouseDown = e => {
      lastX = e.clientX; lastY = e.clientY;
      if (e.button === 0) leftDown = true;
      if (e.button === 2) rightDown = true;
    };
    const onMouseMove = e => {
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      brushX = e.clientX; brushY = e.clientY;
      if (sculptModeRef.current) {
        if (leftDown) applyBrush(e.clientX, e.clientY, -dy * 0.001);
        if (rightDown) { yaw += dx * 0.004; pitch = Math.max(-1.4, Math.min(1.4, pitch - dy * 0.004)); }
        return;
      }
      if (leftDown) {
        yaw   += dx * 0.004;
        pitch  = Math.max(-1.4, Math.min(1.4, pitch - dy * 0.004));
      } else if (rightDown) {
        const r = getCamRight(), u = getCamUp(), s = 0.004;
        camPos[0] -= (dx * r[0] - dy * u[0]) * s;
        camPos[1] -= (dx * r[1] - dy * u[1]) * s;
        camPos[2] -= (dx * r[2] - dy * u[2]) * s;
      }
    };
    const onMouseUp     = e => { if (e.button === 0) leftDown = false; if (e.button === 2) rightDown = false; };
    const onContextMenu = e => e.preventDefault();
    const onWheel = e => {
      const d = getLookDir(), s = e.deltaY * 0.002;
      camPos[0] -= d[0] * s;
      camPos[1] -= d[1] * s;
      camPos[2] -= d[2] * s;
      e.preventDefault();
    };

    const onKeyDown = e => {
      keys.add(e.code);
      if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code)) e.preventDefault();
    };
    const onKeyUp = e => keys.delete(e.code);

    const onTouchStart = e => {
      e.preventDefault();
      if (e.touches.length === 1) {
        leftDown = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        leftDown = false;
        lastTouchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        lastTouchMidX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        lastTouchMidY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      }
    };
    const onTouchMove = e => {
      e.preventDefault();
      if (e.touches.length === 1 && leftDown) {
        const tdx = e.touches[0].clientX - lastX, tdy = e.touches[0].clientY - lastY;
        lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
        if (sculptModeRef.current) { applyBrush(e.touches[0].clientX, e.touches[0].clientY, -tdy * 0.001); brushX = e.touches[0].clientX; brushY = e.touches[0].clientY; return; }
        yaw   += tdx * 0.004;
        pitch  = Math.max(-1.4, Math.min(1.4, pitch - tdy * 0.004));
      } else if (e.touches.length === 2) {
        const t0 = e.touches[0], t1 = e.touches[1];
        // Pinch → move forward/back
        const d2   = Math.hypot(t0.clientX - t1.clientX, t0.clientY - t1.clientY);
        const dir  = getLookDir(), zs = (lastTouchDist - d2) * 0.005;
        camPos[0] += dir[0] * zs; camPos[1] += dir[1] * zs; camPos[2] += dir[2] * zs;
        lastTouchDist = d2;
        // Midpoint drag → pan
        const midX = (t0.clientX + t1.clientX) / 2, midY = (t0.clientY + t1.clientY) / 2;
        const mdx = midX - lastTouchMidX, mdy = midY - lastTouchMidY;
        const r = getCamRight(), u = getCamUp(), ps = 0.004;
        camPos[0] -= (mdx * r[0] - mdy * u[0]) * ps;
        camPos[1] -= (mdx * r[1] - mdy * u[1]) * ps;
        camPos[2] -= (mdx * r[2] - mdy * u[2]) * ps;
        lastTouchMidX = midX; lastTouchMidY = midY;
      }
    };
    const onTouchEnd = () => { leftDown = false; };

    canvas.addEventListener("mousedown",   onMouseDown);
    canvas.addEventListener("contextmenu", onContextMenu);
    canvas.addEventListener("wheel",       onWheel,      { passive: false });
    canvas.addEventListener("touchstart",  onTouchStart, { passive: false });
    canvas.addEventListener("touchmove",   onTouchMove,  { passive: false });
    canvas.addEventListener("touchend",    onTouchEnd);
    window.addEventListener("mousemove",   onMouseMove);
    window.addEventListener("mouseup",     onMouseUp);
    window.addEventListener("keydown",     onKeyDown);
    window.addEventListener("keyup",       onKeyUp);

    // ── Render loop ───────────────────────────────────────────────────────
    let animId;
    function render() {
      animId = requestAnimationFrame(render);

      // Rebuild index buffer if split threshold changed
      if (splitThresholdRef.current !== prevSplitRef.current) {
        rebuildIndices(splitThresholdRef.current);
      }

      // WASD + arrow key movement
      if (keys.size) {
        const spd = 0.025, fwd = getLookDir(), rgt = getCamRight();
        if (keys.has('KeyW')     || keys.has('ArrowUp'))    { camPos[0]+=fwd[0]*spd; camPos[1]+=fwd[1]*spd; camPos[2]+=fwd[2]*spd; }
        if (keys.has('KeyS')     || keys.has('ArrowDown'))  { camPos[0]-=fwd[0]*spd; camPos[1]-=fwd[1]*spd; camPos[2]-=fwd[2]*spd; }
        if (keys.has('KeyA')     || keys.has('ArrowLeft'))  { camPos[0]-=rgt[0]*spd; camPos[1]-=rgt[1]*spd; camPos[2]-=rgt[2]*spd; }
        if (keys.has('KeyD')     || keys.has('ArrowRight')) { camPos[0]+=rgt[0]*spd; camPos[1]+=rgt[1]*spd; camPos[2]+=rgt[2]*spd; }
      }

      gl.clearColor(0.04, 0.02, 0.09, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      const asp = canvas.width / canvas.height;
      const d   = getLookDir();
      const eye = [...camPos];
      const ctr = [eye[0]+d[0], eye[1]+d[1], eye[2]+d[2]];
      const mvp = matMul(perspective(0.85, asp, 0.05, 50), lookAt(eye, ctr, [0, 1, 0]));
      currentMVPRef.current = mvp;

      gl.uniformMatrix4fv(uMVPLoc,   false, mvp);
      gl.uniform1f(uScaleLoc,  depthScaleRef.current);
      gl.uniform1f(uMinLoc,    clipMinRef.current);
      gl.uniform1f(uMaxLoc,    clipMaxRef.current);
      gl.uniform1f(uPSizeLoc,  pointSizeRef.current);
      gl.uniform1f(uSculptActLoc,  sculptModeRef.current ? 1 : 0);
      gl.uniform1f(uEraseModeULoc, eraseModeRef.current  ? 1 : 0);
      const ndcBX = (brushX - canvasRect.left) / canvasRect.width  * 2 - 1;
      const ndcBY = 1 - (brushY - canvasRect.top)  / canvasRect.height * 2;
      gl.uniform2f(uBrushNDCLoc, ndcBX, ndcBY);
      gl.uniform1f(uBrushRadLoc, brushSizeRef.current / canvasRect.width * 2);

      gl.bindBuffer(gl.ARRAY_BUFFER, xyBuf);
      gl.enableVertexAttribArray(aXYLoc);
      gl.vertexAttribPointer(aXYLoc, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, depthBuf);
      gl.enableVertexAttribArray(aDepthLoc);
      gl.vertexAttribPointer(aDepthLoc, 1, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.enableVertexAttribArray(aUVLoc);
      gl.vertexAttribPointer(aUVLoc, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, erasedBuf);
      gl.enableVertexAttribArray(aErasedLoc);
      gl.vertexAttribPointer(aErasedLoc, 1, gl.FLOAT, false, 0, 0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (pointModeRef.current) {
        gl.drawArrays(gl.POINTS, 0, GRID * GRID);
      } else {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
      }
    }
    render();

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      canvas.removeEventListener("mousedown",   onMouseDown);
      canvas.removeEventListener("contextmenu", onContextMenu);
      canvas.removeEventListener("wheel",       onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove",  onTouchMove);
      canvas.removeEventListener("touchend",   onTouchEnd);
      window.removeEventListener("mousemove",  onMouseMove);
      window.removeEventListener("mouseup",    onMouseUp);
      window.removeEventListener("keydown",    onKeyDown);
      window.removeEventListener("keyup",      onKeyUp);
      keys.clear();
      gl.deleteBuffer(xyBuf);
      gl.deleteBuffer(depthBuf);
      gl.deleteBuffer(uvBuf);
      gl.deleteBuffer(erasedBuf);
      gl.deleteBuffer(idxBuf);
      gl.deleteTexture(tex);
      gl.deleteProgram(prog);
      depthsRef.current = null; depthBufRef.current = null; glRef.current = null;
    };
  }, [depthData, depthW, depthH, imageCanvas]);

  // ── Face slider helpers ────────────────────────────────────────────────
  const applyFaceToMesh = useCallback(() => {
    if (!baseDepth || !faceCorrections?.landmarks || !depthsRef.current || !glRef.current) return;
    const GRID  = Math.min(256, depthW, depthH);
    const stepX = (depthW - 1) / (GRID - 1);
    const stepY = (depthH - 1) / (GRID - 1);

    // Reset to pure DepthAnything base
    for (let j = 0; j < GRID; j++)
      for (let i = 0; i < GRID; i++)
        depthsRef.current[j * GRID + i] = baseDepth[Math.round(j * stepY) * depthW + Math.round(i * stepX)];

    // Sculpt-style: each landmark applies a Gaussian stroke to nearby vertices
    // (same principle as the sculpt brush — Gaussian falloff from landmark centre)
    const SIGMA = { nose_tip: 0.030, nose_bridge: 0.018, nose_ala: 0.012, eye: 0.012, mouth: 0.010 };
    const FRAC  = { nose_tip: 0.20,  nose_bridge: 0.10,  nose_ala: 0.07,  eye: 0.07,  mouth: 0.05  };

    // Face mean for scaling
    let dSum = 0, dN = 0;
    for (let k = 0; k < depthsRef.current.length; k++) { dSum += depthsRef.current[k]; dN++; }
    const faceMean = dN > 0 ? dSum / dN : 0.5;

    // Max |corr| per region (to normalise scale)
    const maxCorr = {};
    for (const lm of faceCorrections.landmarks) {
      if (!maxCorr[lm.region] || Math.abs(lm.corr) > maxCorr[lm.region])
        maxCorr[lm.region] = Math.abs(lm.corr);
    }

    for (const lm of faceCorrections.landmarks) {
      const weight = faceSlidersRef.current[lm.region] ?? 0;
      if (weight === 0 || !maxCorr[lm.region]) continue;
      const scale  = faceMean * FRAC[lm.region] / maxCorr[lm.region] * weight;
      const sigmaV = SIGMA[lm.region] * (GRID - 1);
      const sig2   = 2 * sigmaV * sigmaV;
      const rr     = Math.ceil(sigmaV * 2.5);
      const cx = lm.x * (GRID - 1), cy = lm.y * (GRID - 1);
      const x0 = Math.max(0, Math.floor(cx - rr)), x1 = Math.min(GRID - 1, Math.ceil(cx + rr));
      const y0 = Math.max(0, Math.floor(cy - rr)), y1 = Math.min(GRID - 1, Math.ceil(cy + rr));
      for (let vj = y0; vj <= y1; vj++) {
        for (let vi = x0; vi <= x1; vi++) {
          const dx = vi - cx, dy = vj - cy;
          const w  = Math.exp(-(dx * dx + dy * dy) / sig2);
          const idx = vj * GRID + vi;
          depthsRef.current[idx] = Math.max(0, Math.min(1,
            depthsRef.current[idx] + lm.corr * scale * w
          ));
        }
      }
    }

    const gl = glRef.current;
    gl.bindBuffer(gl.ARRAY_BUFFER, depthBufRef.current);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, depthsRef.current);
  }, [baseDepth, faceCorrections, depthW, depthH]);

  const handleFaceSlider = useCallback((key, val) => {
    const s = { ...faceSlidersRef.current, [key]: val };
    faceSlidersRef.current = s;
    setFaceSliders(s);
    applyFaceToMesh();
  }, [applyFaceToMesh]);

  // ── UI ─────────────────────────────────────────────────────────────────
  const row = { display: "flex", alignItems: "center", gap: 12 };
  const lbl = { fontSize: 12, color: "#ccc", width: 44, flexShrink: 0 };
  const val = { fontSize: 11, color: "#00d4ff", width: 38, textAlign: "right", flexShrink: 0 };
  const sld = { flex: 1, accentColor: "#00d4ff", minWidth: 0 };

  return (
    <div style={{ position: "fixed", inset: 0, background: "#060410", zIndex: 150, display: "flex", flexDirection: "column" }}>

      {/* Top bar */}
      <div style={{ padding: "10px 16px", display: "flex", alignItems: "center", gap: 16, background: "rgba(0,0,0,.6)", borderBottom: "1px solid rgba(255,255,255,.15)", flexShrink: 0 }}>
        <span style={{ flex: 1, fontSize: 11, color: "#888", letterSpacing: 1, textTransform: "uppercase" }}>
          3D · vänster: rotera · höger: pan · scroll / nyp: zoom · wasd / ↑↓←→: flytta
        </span>
        <button
          onClick={() => { const v = !pointModeRef.current; pointModeRef.current = v; setPointMode(v); }}
          style={{ padding: "6px 14px", border: `1px solid ${pointMode ? "rgba(0,212,255,.5)" : "rgba(255,255,255,.25)"}`, borderRadius: 8, background: pointMode ? "rgba(0,212,255,.12)" : "rgba(255,255,255,.08)", color: pointMode ? "#00d4ff" : "#ddd", cursor: "pointer", fontFamily: "inherit", fontSize: 12, flexShrink: 0 }}
        >
          ⬡ Punkter
        </button>
        {pointMode && (
          <input
            type="number" min="1" max="20" step="1" value={pointSize}
            onChange={e => { const v = Math.max(1, Math.min(20, parseInt(e.target.value) || 1)); setPointSize(v); pointSizeRef.current = v; }}
            style={{ width: 52, padding: "5px 8px", border: "1px solid rgba(0,212,255,.4)", borderRadius: 8, background: "rgba(0,212,255,.08)", color: "#00d4ff", fontFamily: "inherit", fontSize: 12, textAlign: "center", flexShrink: 0 }}
          />
        )}
        <button
          onClick={() => { const v = !sculptModeRef.current; sculptModeRef.current = v; setSculptMode(v); if (v) setFaceMode(false); }}
          style={{ padding: "6px 14px", border: `1px solid ${sculptMode ? "rgba(0,212,255,.5)" : "rgba(255,255,255,.25)"}`, borderRadius: 8, background: sculptMode ? "rgba(0,212,255,.12)" : "rgba(255,255,255,.08)", color: sculptMode ? "#00d4ff" : "#ddd", cursor: "pointer", fontFamily: "inherit", fontSize: 12, flexShrink: 0 }}
        >
          ✏ Redigera
        </button>
        {faceCorrections?.faceFound && (
          <button
            onClick={() => { setFaceMode(v => { if (!v) { sculptModeRef.current = false; setSculptMode(false); } return !v; }); }}
            style={{ padding: "6px 14px", border: `1px solid ${faceMode ? "rgba(0,212,255,.5)" : "rgba(255,255,255,.25)"}`, borderRadius: 8, background: faceMode ? "rgba(0,212,255,.12)" : "rgba(255,255,255,.08)", color: faceMode ? "#00d4ff" : "#ddd", cursor: "pointer", fontFamily: "inherit", fontSize: 12, flexShrink: 0 }}
          >
            ◎ Ansikte
          </button>
        )}
        <button onClick={onClose} style={{ padding: "6px 14px", border: "1px solid rgba(255,255,255,.25)", borderRadius: 8, background: "rgba(255,255,255,.08)", color: "#ddd", cursor: "pointer", fontFamily: "inherit", fontSize: 12, flexShrink: 0 }}>
          ✕ Stäng
        </button>
      </div>

      {/* Canvas + optional side panel */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
        <canvas ref={canvasRef} style={{ flex: 1, cursor: sculptMode ? "crosshair" : "grab", touchAction: "none" }} />

        {/* Sculpt panel — absolute right side */}
        {sculptMode && (
          <div style={{ position: "absolute", top: 0, right: 0, bottom: 0, width: 176, background: "rgba(6,4,16,.97)", borderLeft: "1px solid rgba(255,255,255,.12)", padding: "16px 14px", display: "flex", flexDirection: "column", gap: 14, overflowY: "auto", zIndex: 2 }}>
            <div style={{ fontSize: 11, color: "#aaa", letterSpacing: 1, textTransform: "uppercase", borderBottom: "1px solid rgba(255,255,255,.08)", paddingBottom: 10 }}>✏ Skulptera</div>
            <div>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>Pensel <span style={{ color: "#00d4ff" }}>{brushSize}px</span></div>
              <input type="range" min="10" max="200" step="5" value={brushSize}
                onChange={e => { const v = +e.target.value; setBrushSize(v); brushSizeRef.current = v; }}
                style={{ width: "100%", accentColor: "#00d4ff" }}
              />
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>Styrka <span style={{ color: "#00d4ff" }}>{brushStrength}</span></div>
              <input type="range" min="1" max="10" step="1" value={brushStrength}
                onChange={e => { const v = +e.target.value; setBrushStrength(v); brushStrengthRef.current = v; }}
                style={{ width: "100%", accentColor: "#00d4ff" }}
              />
            </div>
            <button
              onClick={() => { const v = !eraseModeRef.current; eraseModeRef.current = v; setEraseMode(v); }}
              style={{ padding: "7px 10px", border: `1px solid ${eraseMode ? "rgba(255,80,80,.6)" : "rgba(255,255,255,.15)"}`, borderRadius: 8, background: eraseMode ? "rgba(255,80,80,.15)" : "rgba(255,255,255,.04)", color: eraseMode ? "#ff6060" : "#888", cursor: "pointer", fontFamily: "inherit", fontSize: 11, textAlign: "left" }}
            >
              ✕ Radera punkter
            </button>
            <div style={{ fontSize: 11, color: "#444", lineHeight: 1.8 }}>Dra uppåt: lyfta<br/>Dra nedåt: trycka in</div>
          </div>
        )}

        {/* Face panel — absolute right side */}
        {faceMode && faceCorrections?.faceFound && (
          <div style={{ position: "absolute", top: 0, right: 0, bottom: 0, width: 192, background: "rgba(6,4,16,.97)", borderLeft: "1px solid rgba(255,255,255,.12)", padding: "16px 14px", display: "flex", flexDirection: "column", gap: 14, overflowY: "auto", zIndex: 2 }}>
            <div style={{ fontSize: 11, color: "#aaa", letterSpacing: 1, textTransform: "uppercase", borderBottom: "1px solid rgba(255,255,255,.08)", paddingBottom: 10 }}>◎ Ansikte</div>
            {[
              ["nose_tip",    "Nässpets",   "Hur långt nässpetsen sticker ut"],
              ["nose_bridge", "Näsbrygga",  "Höjd längs näsryggen"],
              ["nose_ala",    "Näsvingar",  "Djup vid näsöppningarna"],
              ["eye",         "Ögon",       "Ögonhålans djup"],
              ["mouth",       "Mun",        "Läppar och mungipa"],
            ].map(([k, label, hint]) => (
              <div key={k}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: "#ccc" }}>{label}</span>
                  <span style={{ fontSize: 11, color: "#00d4ff" }}>{faceSliders[k].toFixed(1)}</span>
                </div>
                <input type="range" min="0" max="5" step="0.1" value={faceSliders[k]}
                  onChange={e => handleFaceSlider(k, parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "#00d4ff" }}
                />
                <div style={{ fontSize: 10, color: "#444", marginTop: 2 }}>{hint}</div>
              </div>
            ))}
            <div style={{ marginTop: "auto", fontSize: 11, color: "#333", lineHeight: 1.8 }}>
              0 = av · 1 = naturligt<br/>2–3 = förstärkt
            </div>
          </div>
        )}

        {/* Mobile sculpt popup */}

      </div>

      {/* Bottom controls */}
      <div style={{ background: "rgba(0,0,0,.6)", borderTop: "1px solid rgba(255,255,255,.15)", flexShrink: 0 }}>

        {showControls && (
          <div style={{ padding: "14px 20px 12px" }}>

            {/* Relief */}
            <div style={{ ...row, marginBottom: 10 }}>
              <span style={lbl}>Relief</span>
              <input type="range" min="0" max="1" step="0.01" value={depthScale} style={sld}
                onChange={e => { const v = parseFloat(e.target.value); setDepthScale(v); depthScaleRef.current = v; }}
              />
              <span style={val}>{depthScale.toFixed(2)}</span>
            </div>

            {/* Kant */}
            <div style={{ ...row, marginBottom: 14 }}>
              <span style={lbl}>Kant</span>
              <input type="range" min="0" max="1" step="0.005" value={splitThreshold} style={sld}
                onChange={e => { const v = parseFloat(e.target.value); setSplitThreshold(v); splitThresholdRef.current = v; }}
              />
              <span style={val}>{splitThreshold.toFixed(3)}</span>
            </div>

            {/* Djupklipp */}
            <div style={{ borderTop: "1px solid rgba(255,255,255,.1)", paddingTop: 12 }}>
              <div style={{ ...row, marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: "#ccc", fontWeight: 600 }}>Djupklipp</span>
                <span style={{ marginLeft: "auto", fontSize: 11, color: "#00d4ff", fontWeight: 600 }}>
                  {Math.round(clipMin * 100)}% – {Math.round(clipMax * 100)}%
                </span>
              </div>
              <div style={row}>
                <span style={{ fontSize: 11, color: "#aaa", width: 60, textAlign: "right", flexShrink: 0 }}>Bakgrund</span>
                <div style={{ flex: 1 }}>
                  <RangeClip minVal={clipMin} maxVal={clipMax} onChange={handleClipChange} />
                </div>
                <span style={{ fontSize: 11, color: "#aaa", width: 60, flexShrink: 0 }}>Förgrund</span>
              </div>
            </div>

          </div>
        )}

        {/* Toggle strip */}
        <button
          onClick={() => setShowControls(v => !v)}
          style={{ width: "100%", padding: "8px", background: "rgba(255,255,255,.04)", border: "none", borderTop: "1px solid rgba(255,255,255,.1)", color: "#aaa", cursor: "pointer", fontFamily: "inherit", fontSize: 11, letterSpacing: 1 }}
        >
          {showControls ? "▼  Dölj kontroller" : "▲  Visa kontroller"}
        </button>

      </div>

    </div>
  );
}
