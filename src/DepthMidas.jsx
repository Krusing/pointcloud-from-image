import { useState, useRef, useCallback, useEffect } from "react";
import { pipeline, env } from "@xenova/transformers";
import Depth3DView from "./Depth3DView";
import { computeFaceCorrections } from "./faceEnhance";

env.allowLocalModels = false;

const PRESETS = [
  { label: "Porträtt", q: "portrait person" },
  { label: "Djur", q: "animals wildlife" },
  { label: "Hundar", q: "dog" },
  { label: "Katter", q: "cat" },
];

const CS = [[0,15,0,50],[.2,30,20,140],[.4,0,100,180],[.6,0,190,190],[.8,220,200,50],[1,255,255,240]];
function colorMap(v) {
  v = v < 0 ? 0 : v > 1 ? 1 : v; let i = 0;
  while (i < CS.length - 2 && CS[i+1][0] < v) i++;
  const t = (v - CS[i][0]) / (CS[i+1][0] - CS[i][0]);
  return [(CS[i][1]+t*(CS[i+1][1]-CS[i][1]))|0,(CS[i][2]+t*(CS[i+1][2]-CS[i][2]))|0,(CS[i][3]+t*(CS[i+1][3]-CS[i][3]))|0];
}

// ── Tiled depth inference ─────────────────────────────────────────────────────
// Runs the model on 4 overlapping quadrants + full image, aligns each tile's
// depth values to the global pass via OLS, then Gaussian-blends into one map.
// Generates NxN tile rects (normalized [0,1] coords) with 50% overlap between neighbours
function generateTiles(N) {
  const tiles = [];
  const sz = (N + 1) / (2 * N);      // tile side length
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++)
      tiles.push([i/(2*N), j/(2*N), i/(2*N)+sz, j/(2*N)+sz]);
  return tiles;
}

async function tiledDepth(pipe, imgEl, outW, outH, N, onProgress) {
  const iw = imgEl.naturalWidth || imgEl.width;
  const ih = imgEl.naturalHeight || imgEl.height;

  const cropUrl = (nx0, ny0, nx1, ny1) => {
    const c = document.createElement('canvas');
    c.width  = Math.round((nx1 - nx0) * iw);
    c.height = Math.round((ny1 - ny0) * ih);
    c.getContext('2d').drawImage(imgEl, Math.round(nx0*iw), Math.round(ny0*ih), c.width, c.height, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.92);
  };

  const runModel = async (url) => {
    const r = await pipe(url);
    const pd = r.predicted_depth || r.depth;
    const dw = pd.dims ? pd.dims[pd.dims.length-1] : pd.width;
    const dh = pd.dims ? pd.dims[pd.dims.length-2] : pd.height;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < pd.data.length; i++) { if(pd.data[i]<mn)mn=pd.data[i]; if(pd.data[i]>mx)mx=pd.data[i]; }
    return { raw: pd.data, dw, dh, mn, rng: (mx-mn)||1 };
  };

  const bilin = (m, nx, ny) => {
    const sx=nx*(m.dw-1), sy=ny*(m.dh-1);
    const x0=sx|0, y0=sy|0, x1=Math.min(m.dw-1,x0+1), y1=Math.min(m.dh-1,y0+1);
    const fx=sx-x0, fy=sy-y0;
    const v=m.raw[y0*m.dw+x0]*(1-fx)*(1-fy)+m.raw[y0*m.dw+x1]*fx*(1-fy)+m.raw[y1*m.dw+x0]*(1-fx)*fy+m.raw[y1*m.dw+x1]*fx*fy;
    return (v-m.mn)/m.rng;
  };

  const total = N * N + 1;
  // Global pass
  onProgress(0, total);
  const G = await runModel(cropUrl(0, 0, 1, 1));
  onProgress(1, total);

  // Seed accumulator with global depth (weight = 1)
  const acc = new Float32Array(outW * outH);
  const wgt = new Float32Array(outW * outH);
  for (let py = 0; py < outH; py++)
    for (let px = 0; px < outW; px++) {
      const idx = py*outW+px;
      acc[idx] = bilin(G, px/(outW-1), py/(outH-1));
      wgt[idx] = 1;
    }

  const TILES = generateTiles(N);

  for (let ti = 0; ti < TILES.length; ti++) {
    const [nx0, ny0, nx1, ny1] = TILES[ti];
    const T = await runModel(cropUrl(nx0, ny0, nx1, ny1));
    onProgress(ti + 2, total);

    const pxMin = Math.max(0, Math.round(nx0*(outW-1)));
    const pxMax = Math.min(outW-1, Math.round(nx1*(outW-1)));
    const pyMin = Math.max(0, Math.round(ny0*(outH-1)));
    const pyMax = Math.min(outH-1, Math.round(ny1*(outH-1)));

    // OLS alignment (sampled every 4th pixel for speed)
    let st=0, sg=0, stt=0, stg=0, n=0;
    for (let py=pyMin; py<=pyMax; py+=4)
      for (let px=pxMin; px<=pxMax; px+=4) {
        const nx=px/(outW-1), ny=py/(outH-1);
        const tn=bilin(T,(nx-nx0)/(nx1-nx0),(ny-ny0)/(ny1-ny0));
        const gn=bilin(G,nx,ny);
        st+=tn; sg+=gn; stt+=tn*tn; stg+=tn*gn; n++;
      }
    let a=1, b=0;
    if (n>1) {
      const tm=st/n, gm=sg/n, den=stt-n*tm*tm;
      if (Math.abs(den)>1e-8) { a=(stg-n*tm*gm)/den; b=gm-a*tm; }
    }

    // Gaussian blend
    const cx=(nx0+nx1)*0.5, cy=(ny0+ny1)*0.5;
    const sig2 = (Math.min(nx1-nx0,ny1-ny0)*0.4)**2;
    for (let py=pyMin; py<=pyMax; py++)
      for (let px=pxMin; px<=pxMax; px++) {
        const nx=px/(outW-1), ny=py/(outH-1);
        const aligned=Math.max(0,Math.min(1, a*bilin(T,(nx-nx0)/(nx1-nx0),(ny-ny0)/(ny1-ny0))+b));
        const dx=nx-cx, dy=ny-cy;
        const w=Math.exp(-(dx*dx+dy*dy)/sig2);
        const idx=py*outW+px;
        acc[idx]+=w*aligned; wgt[idx]+=w;
      }
  }

  const out = new Float32Array(outW*outH);
  for (let i=0; i<out.length; i++) out[i]=wgt[i]>0 ? acc[i]/wgt[i] : 0;
  return out;
}

export default function DepthMidas() {
  const [imageUrl, setImageUrl] = useState(null);
  const [showSplit, setShowSplit] = useState(false);
  const [splitX, setSplitX] = useState(50);
  const [depthStats, setDepthStats] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | loading-model | processing | done | error
  const [loadProgress, setLoadProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [show3D, setShow3D] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalImages, setModalImages] = useState([]);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalSearch, setModalSearch] = useState("");
  const [modalPreset, setModalPreset] = useState(0);
  const [modalHasMore, setModalHasMore] = useState(false);
  const [modalPage, setModalPage] = useState(1);

  const rawDepthRef = useRef(null), depthDimsRef = useRef({ w: 0, h: 0 });
  const depthRef = useRef(null), origRef = useRef(null), splitRef = useRef(null);
  const dispOrigRef = useRef(null), dispDepthRef = useRef(null), splitContainerRef = useRef(null);
  const sentinelRef = useRef(null);
  const activeQueryRef = useRef("");
  const loadMoreFnRef = useRef(null);
  const dragging = useRef(false);
  const pipeRef = useRef(null);
  const [tileGrid,   setTileGrid]   = useState(1);
  const [tileStatus, setTileStatus] = useState(null);
  const tileGridRef = useRef(1);
  const baseDepthRef = useRef(null);
  const faceCorrRef  = useRef(null);

  const copyToDisplay = useCallback(() => {
    [[origRef,dispOrigRef],[depthRef,dispDepthRef]].forEach(([s,d])=>{
      const sc=s.current,dc=d.current; if(!sc||!dc||sc.width===0)return;
      dc.width=sc.width;dc.height=sc.height;dc.getContext("2d").drawImage(sc,0,0);
    });
  }, []);

  const getPipeline = useCallback(async () => {
    if (pipeRef.current) return pipeRef.current;
    setStatus("loading-model"); setLoadProgress(0);
    const device = navigator.gpu ? "webgpu" : "wasm";
    const pipe = await pipeline("depth-estimation", "Xenova/depth-anything-small-hf", {
      device,
      progress_callback: (p) => {
        if (p.status === "progress" && p.progress) setLoadProgress(Math.round(p.progress));
      },
    });
    pipeRef.current = pipe;
    return pipe;
  }, []);

  const processImage = useCallback(async (dataUrl, imgEl) => {
    try {
      const pipe = await getPipeline();
      setStatus("processing"); setTileStatus(null);

      const maxDim = 512;
      let w = imgEl.naturalWidth || imgEl.width, h = imgEl.naturalHeight || imgEl.height;
      const scale = Math.min(1, maxDim / Math.max(w, h));
      w = Math.round(w * scale); h = Math.round(h * scale);

      const oc = origRef.current;
      oc.width = w; oc.height = h;
      oc.getContext("2d").drawImage(imgEl, 0, 0, w, h);

      let interpDepth;
      if (tileGridRef.current > 1) {
        interpDepth = await tiledDepth(pipe, imgEl, w, h, tileGridRef.current,
          (done, total) => setTileStatus(`Körning ${done}/${total}…`));
      } else {
        const result = await pipe(dataUrl);
        const pd = result.predicted_depth || result.depth;
        let depthArray, dw, dh;
        if (pd.dims) { dh=pd.dims[pd.dims.length-2]; dw=pd.dims[pd.dims.length-1]; depthArray=pd.data; }
        else          { dw=pd.width; dh=pd.height; depthArray=pd.data; }
        let min=Infinity, max=-Infinity;
        for (let i=0;i<depthArray.length;i++){if(depthArray[i]<min)min=depthArray[i];if(depthArray[i]>max)max=depthArray[i];}
        const range=max-min||1;
        interpDepth=new Float32Array(w*h);
        for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
          const sx=(x/w)*(dw-1),sy=(y/h)*(dh-1);
          const x0=sx|0,y0=sy|0,x1=Math.min(dw-1,x0+1),y1=Math.min(dh-1,y0+1);
          const fx=sx-x0,fy=sy-y0;
          const raw=depthArray[y0*dw+x0]*(1-fx)*(1-fy)+depthArray[y0*dw+x1]*fx*(1-fy)+depthArray[y1*dw+x0]*(1-fx)*fy+depthArray[y1*dw+x1]*fx*fy;
          interpDepth[y*w+x]=(raw-min)/range;
        }
      }

      // Face landmark detection — stores geometry for 3D sculpt sliders, does NOT modify depth map
      setTileStatus('Söker ansiktsgeometri…');
      let faceFound = false;
      baseDepthRef.current = interpDepth;
      faceCorrRef.current  = null;
      try {
        const corrections = await computeFaceCorrections(imgEl, w, h);
        if (corrections.faceFound) {
          faceCorrRef.current = corrections;
          faceFound = true;
        }
      } catch (e) {
        console.warn('Face detection skipped:', e);
      }

      const dc = depthRef.current;
      dc.width = w; dc.height = h;
      const ctx = dc.getContext("2d"), imd = ctx.createImageData(w, h), px = imd.data;
      let sum = 0, fg = 0;
      for (let i = 0; i < w*h; i++) {
        const v = interpDepth[i];
        const [r,g,b] = colorMap(v);
        px[i*4]=r; px[i*4+1]=g; px[i*4+2]=b; px[i*4+3]=255;
        sum+=v; if(v>.5)fg++;
      }
      ctx.putImageData(imd, 0, 0);
      rawDepthRef.current = interpDepth; depthDimsRef.current = { w, h };

      setDepthStats({ avgDepth:((sum/(w*h))*100).toFixed(1), fgPercent:((fg/(w*h))*100).toFixed(1), resolution:`${w}×${h}`, faceFound });
      setTileStatus(null);
      setTimeout(copyToDisplay, 20);
      setStatus("done");
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Okänt fel");
      setStatus("error");
    }
  }, [getPipeline, copyToDisplay]);

  const searchPexels = useCallback(async (q, page = 1) => {
    if (!q.trim()) return;
    if (page === 1) { activeQueryRef.current = q; setModalImages([]); }
    setModalLoading(true);
    try {
      const res = await fetch(
        `https://api.pexels.com/v1/search?query=${encodeURIComponent(q)}&per_page=24&page=${page}`,
        { headers: { Authorization: import.meta.env.VITE_PEXELS_KEY } }
      );
      const data = await res.json();
      const imgs = (data.photos || []).map(p => ({
        title: p.alt || q,
        thumb: p.src.medium,
        full: p.src.large,
        photographer: p.photographer,
      }));
      setModalImages(prev => page === 1 ? imgs : [...prev, ...imgs]);
      setModalPage(page);
      setModalHasMore(!!data.next_page);
    } catch(e) {
      console.error(e);
    } finally {
      setModalLoading(false);
    }
  }, []);

  const openModal = useCallback(() => {
    setShowModal(true);
    setModalPreset(0);
    searchPexels(PRESETS[0].q);
  }, [searchPexels]);

  // uppdatera load-more-funktionen varje render
  loadMoreFnRef.current = () => {
    if (!modalLoading && modalHasMore) searchPexels(activeQueryRef.current, modalPage + 1);
  };

  useEffect(() => {
    if (!showModal) return;
    const el = sentinelRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) loadMoreFnRef.current?.(); },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [showModal, modalImages]);

  const pickImage = useCallback((img) => {
    setShowModal(false);
    setImageUrl(img.thumb);
    setErrorMsg("");
    setStatus("processing");
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => {
      const scale = Math.min(1, 1200 / Math.max(el.naturalWidth, el.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.round(el.naturalWidth * scale);
      c.height = Math.round(el.naturalHeight * scale);
      c.getContext("2d").drawImage(el, 0, 0, c.width, c.height);
      const dataUrl = c.toDataURL("image/jpeg", 0.92);
      processImage(dataUrl, el);
    };
    el.onerror = () => { setErrorMsg("Kunde inte ladda bilden från Pexels"); setStatus("error"); };
    el.src = img.full;
  }, [processImage]);

  // Split view
  const drawSplit = useCallback((pct) => {
    const s=splitRef.current,o=origRef.current,d=depthRef.current;
    if(!s||!o||!d||o.width===0)return; const w=o.width,h=o.height;
    s.width=w;s.height=h;const ctx=s.getContext("2d"),cx=Math.round((pct/100)*w);
    ctx.clearRect(0,0,w,h);ctx.drawImage(o,0,0,cx,h,0,0,cx,h);ctx.drawImage(d,cx,0,w-cx,h,cx,0,w-cx,h);
  }, []);

  useEffect(()=>{if(showSplit&&status==="done")drawSplit(splitX);},[showSplit,splitX,drawSplit,status,depthStats]);
  useEffect(()=>{if(!showSplit&&status==="done")setTimeout(copyToDisplay,20);},[showSplit,status,copyToDisplay]);

  const handleFile=(f)=>{if(!f||!f.type.startsWith("image/"))return;const r=new FileReader();r.onload=(e)=>{const url=e.target.result;const img=new Image();img.onload=()=>{setImageUrl(url);setErrorMsg("");setStatus("processing");processImage(url,img);};img.src=url;};r.readAsDataURL(f);};
  const handleDrop=(e)=>{e.preventDefault();e.stopPropagation();handleFile(e.dataTransfer?.files?.[0]);};
  const handleSplitMove=useCallback((e)=>{if(!dragging.current&&e.type!=="mousedown"&&e.type!=="touchstart")return;const rect=splitContainerRef.current?.getBoundingClientRect();if(!rect)return;const x=(e.clientX??e.touches?.[0]?.clientX)-rect.left;setSplitX(Math.max(2,Math.min(98,(x/rect.width)*100)));}, []);
  const reset=()=>{
    setImageUrl(null);setDepthStats(null);setShowSplit(false);setStatus("idle");setErrorMsg("");
    baseDepthRef.current=null;faceCorrRef.current=null;
  };

  useEffect(()=>{
    const onMove=(e)=>{if(dragging.current)handleSplitMove(e);};
    const onUp=()=>{dragging.current=false;};
    window.addEventListener("mousemove",onMove);
    window.addEventListener("mouseup",onUp);
    return()=>{window.removeEventListener("mousemove",onMove);window.removeEventListener("mouseup",onUp);};
  },[handleSplitMove]);

  const isLoading = status==="loading-model"||status==="processing";

  return (
    <div style={{padding:24,overflow:"hidden"}}>
      <style>{`
        html,body{height:100%;overflow:hidden;margin:0}
        .gp{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;backdrop-filter:blur(12px)}
        .mb{padding:8px 16px;border:1px solid rgba(255,255,255,.1);border-radius:8px;background:rgba(255,255,255,.04);color:#9995a8;cursor:pointer;font-family:inherit;font-size:12px;transition:all .2s}
        .mb:hover{background:rgba(255,255,255,.08);color:#ccc}
        .mb.a{background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(123,97,255,.15));border-color:rgba(0,212,255,.4);color:#00d4ff}
        .dz{border:2px dashed rgba(255,255,255,.12);border-radius:16px;padding:48px 24px;text-align:center;cursor:pointer;transition:all .3s}
        .dz:hover,.dz.dg{border-color:rgba(0,212,255,.4);background:rgba(0,212,255,.04)}
        .sp{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:20px;font-size:11px;color:#8884a0}
        .sv{color:#00d4ff;font-weight:600}
        canvas{border-radius:8px;display:block}
        .ic{width:auto;max-width:100%;max-height:calc(100vh - 248px)}
        .sc{width:auto;max-width:100%;max-height:calc(100vh - 248px)}
        .cl{position:absolute;top:8px;padding:4px 10px;background:rgba(0,0,0,.6);border-radius:6px;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#9995a8;z-index:4}
        .pbar{height:4px;border-radius:2px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:12px;max-width:320px}
        .pfill{height:100%;border-radius:2px;background:linear-gradient(90deg,#00d4ff,#7b61ff);transition:width .3s}
        @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
        .modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:200;backdrop-filter:blur(6px)}
        .modal-panel{background:#0e0c16;border:1px solid rgba(255,255,255,.1);border-radius:16px;width:min(760px,95vw);max-height:85vh;display:flex;flex-direction:column;overflow:hidden}
        .th{border-radius:8px;overflow:hidden;cursor:pointer;aspect-ratio:1;border:2px solid transparent;transition:border-color .15s}
        .th:hover{border-color:#00d4ff}
        .th img{width:100%;height:100%;object-fit:cover;display:block}
        .mi{flex:1;padding:6px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:8px;color:#ccc;font-size:12px;outline:none;font-family:inherit}
        .mi:focus{border-color:rgba(0,212,255,.4)}
        .sk{border-radius:8px;aspect-ratio:1;background:rgba(255,255,255,.04);animation:pulse 1.5s ease infinite}
      `}</style>

      <div style={{maxWidth:880,margin:"0 auto"}}>
        <div style={{marginBottom:24}}>
          <div style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:28,fontWeight:800,color:"#ffffff"}}>
            Djupuppskattning — Neuralt nätverk
          </div>
          <p style={{color:"#6b6580",fontSize:13,marginTop:6}}>Neuralt nätverk uppskattar avstånd per pixel — körs lokalt i din webbläsare</p>
        </div>

        {/* View controls */}
        {status==="done"&&(
          <div className="gp" style={{padding:16,marginBottom:16}}>
            <div style={{display:"flex",gap:8,alignItems:"center"}}>
              <span style={{fontSize:11,color:"#6b6580",textTransform:"uppercase",letterSpacing:1}}>Vy:</span>
              <button className={`mb ${!showSplit?"a":""}`} onClick={()=>setShowSplit(false)}>⊞ Sida vid sida</button>
              <button className={`mb ${showSplit?"a":""}`} onClick={()=>setShowSplit(true)}>⊟ Jämför</button>
              <button className="mb" onClick={()=>setShow3D(true)}>◈ 3D</button>
            </div>
          </div>
        )}

        {/* Upload zone */}
        {!imageUrl?(
          <div className={`dz ${dragOver?"dg":""}`} onDrop={e=>{handleDrop(e);setDragOver(false)}} onDragOver={e=>{e.preventDefault();setDragOver(true)}} onDragLeave={()=>setDragOver(false)} onClick={()=>document.getElementById("mf-gallery").click()}>
            <input id="mf-gallery" type="file" accept="image/*" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
            <input id="mf-camera"  type="file" accept="image/*" capture="environment" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" style={{marginBottom:16,opacity:dragOver?.9:.35}}><rect x="8" y="12" width="48" height="40" rx="6" stroke="#7b61ff" strokeWidth="2" fill="none"/><circle cx="22" cy="28" r="5" stroke="#00d4ff" strokeWidth="1.5" fill="none"/><path d="M8 42 L22 32 L34 40 L44 30 L56 42" stroke="#ff6ec7" strokeWidth="1.5" fill="none"/><path d="M32 4 L32 18 M26 12 L32 6 L38 12" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <div style={{fontSize:15,color:dragOver?"#00d4ff":"#8b84a0",marginBottom:6,fontFamily:"'Space Grotesk',sans-serif",fontWeight:600}}>{dragOver?"Släpp bilden här!":"Välj en bild"}</div>
            <div style={{fontSize:12,color:"#5a5470",marginBottom:12}}>Dra hit eller klicka för att öppna galleri</div>
            <div style={{fontSize:11,color:"#6b6580",maxWidth:400,margin:"16px auto 0",lineHeight:1.8}}>
              Djupuppskattningsmodellen (~97 MB) laddas vid första bilden och cachas sedan i webbläsaren
            </div>
            <div style={{display:"flex",alignItems:"center",gap:12,maxWidth:400,margin:"20px auto 0"}}>
              <div style={{flex:1,height:1,background:"rgba(255,255,255,.06)"}}/>
              <span style={{fontSize:11,color:"#4a4560"}}>eller</span>
              <div style={{flex:1,height:1,background:"rgba(255,255,255,.06)"}}/>
            </div>
            <div style={{display:"flex",gap:8,marginTop:12,justifyContent:"center",flexWrap:"wrap"}}>
              <button className="mb" onClick={e=>{e.stopPropagation();document.getElementById("mf-camera").click();}}>
                ◎ Kamera
              </button>
              <button className="mb" onClick={e=>{e.stopPropagation();openModal();}}>
                ◈ Pexels
              </button>
            </div>
            <div onClick={e=>e.stopPropagation()} style={{display:"flex",alignItems:"center",gap:8,justifyContent:"center",marginTop:16}}>
              <span style={{fontSize:12,color:"#6b6580"}}>Detalj-rutnät</span>
              <input type="number" min="1" max="4" step="1" value={tileGrid}
                onChange={e=>{const v=Math.max(1,Math.min(4,parseInt(e.target.value)||1));setTileGrid(v);tileGridRef.current=v;}}
                style={{width:44,padding:"4px 8px",border:`1px solid ${tileGrid>1?"rgba(0,212,255,.4)":"rgba(255,255,255,.15)"}`,borderRadius:8,background:"rgba(255,255,255,.04)",color:tileGrid>1?"#00d4ff":"#aaa",fontFamily:"inherit",fontSize:12,textAlign:"center"}}
              />
              <span style={{fontSize:11,color:"#4a4560"}}>{tileGrid<=1?"1 körning · snabbt":`${tileGrid*tileGrid+1} körningar · ~${tileGrid*tileGrid+1}× längre`}</span>
            </div>
          </div>
        ):(
          <div>
            {/* Loading */}
            {isLoading&&(
              <div className="gp" style={{padding:32,marginBottom:16,textAlign:"center"}}>
                <div style={{fontSize:13,color:"#8b84a0",animation:"pulse 1.5s ease infinite"}}>
                  {status==="loading-model"?`Laddar djupuppskattningsmodell... ${loadProgress}%`:tileStatus||"Analyserar djup med neuralt nätverk..."}
                </div>
                {status==="loading-model"&&<div className="pbar" style={{margin:"12px auto 0"}}><div className="pfill" style={{width:`${loadProgress}%`}}/></div>}
                <div style={{fontSize:11,color:"#4a4560",marginTop:12}}>
                  {status==="loading-model"?"Första gången tar längst — modellen cachas i webbläsaren":"Neuralt nätverk beräknar djup per pixel..."}
                </div>
              </div>
            )}

            {/* Error */}
            {status==="error"&&(
              <div className="gp" style={{padding:24,marginBottom:16,borderColor:"rgba(255,100,100,.2)"}}>
                <div style={{color:"#ff6b6b",fontSize:13,marginBottom:8}}>Något gick fel</div>
                <div style={{color:"#6b6580",fontSize:11,marginBottom:12}}>{errorMsg}</div>
                <button className="mb" onClick={reset} style={{borderColor:"rgba(255,100,100,.3)",color:"#ff6b6b"}}>Försök igen</button>
              </div>
            )}

            {/* Images + Stats sidebar */}
            {status==="done"&&(
              <div style={{display:"flex",gap:12,alignItems:"flex-start"}}>
                {/* Images */}
                <div style={{flex:1,minWidth:0}}>
                  {showSplit?(
                    <div className="gp" style={{padding:4,display:"flex",justifyContent:"center"}}>
                      <div ref={splitContainerRef} style={{position:"relative",userSelect:"none",cursor:"col-resize",display:"inline-block"}}
                        onMouseDown={e=>{dragging.current=true;handleSplitMove(e);}}
                        onTouchStart={e=>{dragging.current=true;handleSplitMove(e);}} onTouchMove={handleSplitMove} onTouchEnd={()=>{dragging.current=false;}}>
                        <canvas ref={splitRef} className="sc"/>
                        <div style={{position:"absolute",top:0,bottom:0,left:`${splitX}%`,width:2,marginLeft:-1,background:"#fff",boxShadow:"0 0 8px rgba(0,0,0,.6)",pointerEvents:"none",zIndex:2}}/>
                        <div style={{position:"absolute",top:"50%",left:`${splitX}%`,transform:"translate(-50%,-50%)",width:32,height:32,borderRadius:"50%",background:"rgba(255,255,255,.95)",boxShadow:"0 2px 12px rgba(0,0,0,.4)",display:"flex",alignItems:"center",justifyContent:"center",pointerEvents:"none",zIndex:3}}>
                          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 3L1 8L5 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M11 3L15 8L11 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                        </div>
                        <div className="cl" style={{left:8}}>ORIGINAL</div>
                        <div className="cl" style={{right:8,left:"auto"}}>DJUP</div>
                      </div>
                    </div>
                  ):(
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                      <div className="gp" style={{padding:4,position:"relative"}}><canvas ref={dispOrigRef} className="ic"/><div className="cl" style={{left:8}}>ORIGINAL</div></div>
                      <div className="gp" style={{padding:4,position:"relative"}}><canvas ref={dispDepthRef} className="ic"/><div className="cl" style={{left:8}}>DJUP</div></div>
                    </div>
                  )}
                  <div style={{marginTop:12,display:"flex",alignItems:"center",gap:8}}>
                    <span style={{fontSize:10,color:"#4a4560"}}>NÄRA</span>
                    <div style={{flex:1,height:8,borderRadius:4,background:"linear-gradient(90deg,#f0e6b8,#dcc832,#00bebe,#1e64b4,#0f0032)"}}/>
                    <span style={{fontSize:10,color:"#4a4560"}}>LÅNGT</span>
                  </div>
                </div>

                {/* Stats sidebar */}
                {depthStats&&(
                  <div className="gp" style={{padding:12,display:"flex",flexDirection:"column",gap:8,minWidth:160}}>
                    <span className="sp">Upplösning <span className="sv">{depthStats.resolution}</span></span>
                    <span className="sp">Medeldjup <span className="sv">{depthStats.avgDepth}%</span></span>
                    <span className="sp">Förgrund <span className="sv">{depthStats.fgPercent}%</span></span>
                    {depthStats.faceFound && <span className="sp">Ansikte <span className="sv">↑ 3D</span></span>}
                    <div style={{flex:1}}/>
                    <button className="mb" onClick={reset} style={{fontSize:11,marginTop:8}}>✕ Ny bild</button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

      </div>

      {/* Persistent hidden canvases */}
      <div style={{position:"absolute",left:-9999,top:-9999}}><canvas ref={origRef}/><canvas ref={depthRef}/></div>

      {show3D&&rawDepthRef.current&&(
        <Depth3DView
          depthData={rawDepthRef.current}
          depthW={depthDimsRef.current.w}
          depthH={depthDimsRef.current.h}
          imageCanvas={origRef.current}
          onClose={()=>setShow3D(false)}
          baseDepth={baseDepthRef.current}
          faceCorrections={faceCorrRef.current}
        />
      )}

      {/* Commons modal */}
      {showModal&&(
        <div className="modal-ov" onClick={()=>setShowModal(false)}>
          <div className="modal-panel" onClick={e=>e.stopPropagation()}>

            {/* Header */}
            <div style={{padding:"16px 20px",borderBottom:"1px solid rgba(255,255,255,.08)",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
              <span style={{fontSize:14,fontWeight:600,color:"#ccc",fontFamily:"'Space Grotesk',sans-serif"}}>Välj ett foto</span>
              <button className="mb" style={{padding:"4px 10px"}} onClick={()=>setShowModal(false)}>✕</button>
            </div>

            {/* Filters + search */}
            <div style={{padding:"12px 16px",borderBottom:"1px solid rgba(255,255,255,.06)",display:"flex",gap:8,flexWrap:"wrap",alignItems:"center",flexShrink:0}}>
              {PRESETS.map((p,i)=>(
                <button key={i} className={`mb${modalPreset===i?" a":""}`}
                  onClick={()=>{setModalPreset(i);setModalSearch("");searchPexels(p.q);}}>
                  {p.label}
                </button>
              ))}
              <div style={{display:"flex",gap:6,flex:1,minWidth:180}}>
                <input className="mi" value={modalSearch} onChange={e=>setModalSearch(e.target.value)}
                  onKeyDown={e=>{if(e.key==="Enter"&&modalSearch.trim()){setModalPreset(-1);searchPexels(modalSearch);}}}
                  placeholder="Sök egna bilder…"/>
                <button className="mb" onClick={()=>{if(modalSearch.trim()){setModalPreset(-1);searchPexels(modalSearch);}}}>Sök</button>
              </div>
            </div>

            {/* Grid */}
            <div style={{padding:16,overflowY:"auto",flex:1}}>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
                {modalLoading&&modalImages.length===0
                  ? Array.from({length:16}).map((_,i)=><div key={i} className="sk" style={{animationDelay:`${i*0.04}s`}}/>)
                  : modalImages.length===0
                    ? <div style={{gridColumn:"1/-1",textAlign:"center",padding:40,color:"#5a5470",fontSize:12}}>Inga bilder hittades</div>
                    : modalImages.map((img,i)=>(
                        <div key={i} className="th" onClick={()=>pickImage(img)} title={img.title}>
                          <img src={img.thumb} alt={img.title} loading="lazy" crossOrigin="anonymous"/>
                        </div>
                      ))
                }
              </div>
              <div ref={sentinelRef} style={{height:40,display:"flex",alignItems:"center",justifyContent:"center"}}>
                {modalLoading&&modalImages.length>0&&<div style={{fontSize:11,color:"#4a4560",animation:"pulse 1.5s ease infinite"}}>Laddar fler…</div>}
              </div>
            </div>

            {/* Footer */}
            <div style={{padding:"10px 20px",borderTop:"1px solid rgba(255,255,255,.06)",fontSize:10,color:"#3a3550",flexShrink:0}}>
              Pexels · Gratis att använda · Foto av {modalImages[0]?.photographer ?? "…"} m.fl.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
