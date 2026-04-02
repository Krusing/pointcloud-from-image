import { useState, useRef, useCallback, useEffect } from "react";
import Depth3DView from "./Depth3DView";

const PRESETS = [
  { label: "Porträtt", q: "portrait person" },
  { label: "Djur", q: "animals wildlife" },
  { label: "Hundar", q: "dog" },
  { label: "Katter", q: "cat" },
];

const METHODS = {
  luminance: { label: "Luminans", desc: "Ljusare = närmare" },
  invLuminance: { label: "Inv. Luminans", desc: "Mörkare = närmare" },
  saturation: { label: "Mättnad", desc: "Mer mättad = närmare" },
  combined: { label: "Kombinerad", desc: "Skärpa + position + mättnad + kontrast" },
};

function boxBlur(src, w, h, radius) {
  if (radius < 1) return new Float32Array(src);
  const r = Math.round(radius), tmp = new Float32Array(w * h), out = new Float32Array(w * h), diam = r * 2 + 1;
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[y * w + Math.max(0, Math.min(w - 1, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = sum / diam;
      sum += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.max(0, Math.min(h - 1, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / diam;
      sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}
function fastBlur(s, w, h, r) { if (r < 1) return new Float32Array(s); let d = boxBlur(s, w, h, r); d = boxBlur(d, w, h, r); return boxBlur(d, w, h, r); }

function localContrast(gray, w, h, bs) {
  const out = new Float32Array(w * h), half = Math.floor(bs / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let mn = 1, mx = 0;
    for (let dy = -half; dy <= half; dy += 3) for (let dx = -half; dx <= half; dx += 3) {
      const v = gray[Math.max(0, Math.min(h - 1, y + dy)) * w + Math.max(0, Math.min(w - 1, x + dx))];
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    out[y * w + x] = mx - mn;
  }
  return out;
}
function edgeSharpness(gray, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const tl=gray[(y-1)*w+(x-1)],tc=gray[(y-1)*w+x],tr=gray[(y-1)*w+(x+1)];
    const ml=gray[y*w+(x-1)],mr=gray[y*w+(x+1)];
    const bl=gray[(y+1)*w+(x-1)],bc=gray[(y+1)*w+x],br=gray[(y+1)*w+(x+1)];
    const gx=-tl-2*ml-bl+tr+2*mr+br, gy=-tl-2*tc-tr+bl+2*bc+br;
    out[y*w+x]=Math.sqrt(gx*gx+gy*gy);
  }
  return out;
}
function verticalGradient(w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) { const v = y / (h - 1); for (let x = 0; x < w; x++) out[y * w + x] = v; }
  return out;
}
function centerBias(w, h) {
  const out = new Float32Array(w * h), cx = w / 2, cy = h / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = (x - cx) / cx, dy = (y - cy) / cy;
    out[y * w + x] = 1 - Math.sqrt(dx * dx + dy * dy) / Math.SQRT2;
  }
  return out;
}

const CS = [[0,15,0,50],[.2,30,20,140],[.4,0,100,180],[.6,0,190,190],[.8,220,200,50],[1,255,255,240]];
function colorMap(v) {
  v = v < 0 ? 0 : v > 1 ? 1 : v; let i = 0;
  while (i < CS.length - 2 && CS[i + 1][0] < v) i++;
  const t = (v - CS[i][0]) / (CS[i + 1][0] - CS[i][0]);
  return [(CS[i][1]+t*(CS[i+1][1]-CS[i][1]))|0,(CS[i][2]+t*(CS[i+1][2]-CS[i][2]))|0,(CS[i][3]+t*(CS[i+1][3]-CS[i][3]))|0];
}
function normalize(a) {
  let mn=Infinity,mx=-Infinity;
  for (let i=0;i<a.length;i++){if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}
  const r=mx-mn||1,o=new Float32Array(a.length);
  for(let i=0;i<a.length;i++)o[i]=(a[i]-mn)/r;return o;
}

export default function DepthHeuristic() {
  const [image, setImage] = useState(null);
  const [show3D, setShow3D] = useState(false);
  const [method, setMethod] = useState("combined");
  const [blur, setBlur] = useState(3);
  const [showSplit, setShowSplit] = useState(false);
  const [splitX, setSplitX] = useState(50);
  const [depthStats, setDepthStats] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalImages, setModalImages] = useState([]);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalSearch, setModalSearch] = useState("");
  const [modalPreset, setModalPreset] = useState(0);
  const [modalHasMore, setModalHasMore] = useState(false);
  const [modalPage, setModalPage] = useState(1);

  const depthRef = useRef(null), origRef = useRef(null), splitRef = useRef(null);
  const dispOrigRef = useRef(null), dispDepthRef = useRef(null), splitContainerRef = useRef(null);
  const dragging = useRef(false), rawDepthRef = useRef(null), dimRef = useRef({ w: 0, h: 0 });
  const sentinelRef = useRef(null);
  const activeQueryRef = useRef("");
  const loadMoreFnRef = useRef(null);

  const copyToDisplay = useCallback(() => {
    [[origRef,dispOrigRef],[depthRef,dispDepthRef]].forEach(([s,d])=>{
      const sc=s.current,dc=d.current; if(!sc||!dc||sc.width===0)return;
      dc.width=sc.width;dc.height=sc.height;dc.getContext("2d").drawImage(sc,0,0);
    });
  }, []);

  const renderDepth = useCallback((raw, w, h, bv) => {
    const bl = normalize(fastBlur(raw, w, h, bv)), dc = depthRef.current;
    if (!dc) return; dc.width = w; dc.height = h;
    const ctx = dc.getContext("2d"), img = ctx.createImageData(w, h), px = img.data;
    let sum = 0, fg = 0;
    for (let i = 0; i < w * h; i++) {
      const [r,g,b] = colorMap(bl[i]); const j = i * 4;
      px[j]=r;px[j+1]=g;px[j+2]=b;px[j+3]=255; sum+=bl[i]; if(bl[i]>.5)fg++;
    }
    ctx.putImageData(img, 0, 0);
    setDepthStats({ avgDepth:((sum/(w*h))*100).toFixed(1), fgPercent:((fg/(w*h))*100).toFixed(1), resolution:`${w}×${h}` });
    setTimeout(copyToDisplay, 10);
  }, [copyToDisplay]);

  const computeRawDepth = useCallback((imgEl, m) => {
    const maxDim = 480; let w = imgEl.naturalWidth, h = imgEl.naturalHeight;
    const scale = Math.min(1, maxDim / Math.max(w, h));
    w = Math.round(w * scale); h = Math.round(h * scale); dimRef.current = { w, h };
    const off = document.createElement("canvas"); off.width = w; off.height = h;
    const ctx = off.getContext("2d"); ctx.drawImage(imgEl, 0, 0, w, h);
    const pixels = ctx.getImageData(0, 0, w, h).data;
    const oc = origRef.current;
    if (oc) { oc.width = w; oc.height = h; oc.getContext("2d").drawImage(imgEl, 0, 0, w, h); }
    const n = w * h, gray = new Float32Array(n);
    for (let i = 0; i < n; i++) { const j = i * 4; gray[i] = (0.299*pixels[j]+0.587*pixels[j+1]+0.114*pixels[j+2])/255; }
    let depth;
    if (m === "combined") {
      const sat = new Float32Array(n);
      for (let i=0;i<n;i++){const j=i*4;const mx=Math.max(pixels[j],pixels[j+1],pixels[j+2])/255;const mn=Math.min(pixels[j],pixels[j+1],pixels[j+2])/255;sat[i]=mx===0?0:(mx-mn)/mx;}
      const contrast = localContrast(gray,w,h,9), sharpness = normalize(fastBlur(edgeSharpness(gray,w,h),w,h,3));
      const vGrad = verticalGradient(w,h), cBias = centerBias(w,h);
      depth = new Float32Array(n);
      for(let i=0;i<n;i++) depth[i]=.30*sharpness[i]+.25*vGrad[i]+.15*cBias[i]+.10*gray[i]+.10*sat[i]+.10*contrast[i];
    } else if (m==="luminance") { depth=new Float32Array(gray);
    } else if (m==="invLuminance") { depth=new Float32Array(n); for(let i=0;i<n;i++)depth[i]=1-gray[i];
    } else { depth=new Float32Array(n); for(let i=0;i<n;i++){const j=i*4;const mx=Math.max(pixels[j],pixels[j+1],pixels[j+2])/255;const mn=Math.min(pixels[j],pixels[j+1],pixels[j+2])/255;depth[i]=mx===0?0:(mx-mn)/mx;} }
    return normalize(depth);
  }, []);

  const processImage = useCallback((img, m, b) => {
    if (!img) return; const raw = computeRawDepth(img, m); rawDepthRef.current = raw;
    renderDepth(raw, dimRef.current.w, dimRef.current.h, b);
  }, [computeRawDepth, renderDepth]);

  const handleBlurChange = useCallback((b) => {
    if (!rawDepthRef.current) return; renderDepth(rawDepthRef.current, dimRef.current.w, dimRef.current.h, b);
  }, [renderDepth]);

  const drawSplit = useCallback((pct) => {
    const s=splitRef.current,o=origRef.current,d=depthRef.current;
    if(!s||!o||!d||o.width===0)return; const w=o.width,h=o.height;
    s.width=w;s.height=h;const ctx=s.getContext("2d"),cx=Math.round((pct/100)*w);
    ctx.clearRect(0,0,w,h);ctx.drawImage(o,0,0,cx,h,0,0,cx,h);ctx.drawImage(d,cx,0,w-cx,h,cx,0,w-cx,h);
  }, []);

  useEffect(()=>{if(showSplit)drawSplit(splitX);},[showSplit,splitX,drawSplit,depthStats]);
  useEffect(()=>{if(!showSplit&&image)setTimeout(copyToDisplay,20);},[showSplit,image,copyToDisplay]);

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
        full: p.src.medium,
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
    const el = new Image();
    el.crossOrigin = "anonymous";
    el.onload = () => {
      setImage(el);
      setDepthStats(null);
      setShowSplit(false);
      rawDepthRef.current = null;
      setTimeout(() => processImage(el, method, blur), 30);
    };
    el.src = img.full;
  }, [processImage, method, blur]);

  const handleFile=(f)=>{if(!f||!f.type.startsWith("image/"))return;const r=new FileReader();r.onload=(e)=>{const img=new Image();img.onload=()=>{setImage(img);setTimeout(()=>processImage(img,method,blur),30);};img.src=e.target.result;};r.readAsDataURL(f);};
  const handleDrop=(e)=>{e.preventDefault();e.stopPropagation();handleFile(e.dataTransfer?.files?.[0]);};
  const handleSplitMove=(e)=>{if(!dragging.current&&e.type!=="mousedown"&&e.type!=="touchstart")return;const rect=splitContainerRef.current?.getBoundingClientRect();if(!rect)return;const x=(e.clientX??e.touches?.[0]?.clientX)-rect.left;setSplitX(Math.max(2,Math.min(98,(x/rect.width)*100)));};

  useEffect(()=>{
    const onMove=(e)=>{if(dragging.current)handleSplitMove(e);};
    const onUp=()=>{dragging.current=false;};
    window.addEventListener("mousemove",onMove);
    window.addEventListener("mouseup",onUp);
    return()=>{window.removeEventListener("mousemove",onMove);window.removeEventListener("mouseup",onUp);};
  },[]);

  return (
    <div style={{color:"#e0dfe6",fontFamily:"'JetBrains Mono','Fira Code','SF Mono',monospace",padding:24}}>
      <style>{`
        html,body{height:100%;overflow:hidden;margin:0}
        *{box-sizing:border-box;margin:0;padding:0}
        .dt{font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:#ffffff}
        .gp{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;backdrop-filter:blur(12px)}
        .mb{padding:8px 16px;border:1px solid rgba(255,255,255,.1);border-radius:8px;background:rgba(255,255,255,.04);color:#9995a8;cursor:pointer;font-family:inherit;font-size:12px;transition:all .2s}
        .mb:hover{background:rgba(255,255,255,.08);color:#ccc}
        .mb.a{background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(123,97,255,.15));border-color:rgba(0,212,255,.4);color:#00d4ff}
        .dz{border:2px dashed rgba(255,255,255,.12);border-radius:16px;padding:48px 24px;text-align:center;cursor:pointer;transition:all .3s;position:relative;overflow:hidden}
        .dz:hover,.dz.dg{border-color:rgba(0,212,255,.4);background:rgba(0,212,255,.04)}
        .st{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;background:rgba(255,255,255,.1);outline:none}
        .st::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:#00d4ff;cursor:grab}
        .sp{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:20px;font-size:11px;color:#8884a0}
        .sv{color:#00d4ff;font-weight:600}
        canvas{border-radius:8px;display:block}
        .ic{width:auto;max-width:100%;max-height:calc(100vh - 295px)}
        .sc{width:auto;max-width:100%;max-height:calc(100vh - 295px)}
        .cl{position:absolute;top:8px;padding:4px 10px;background:rgba(0,0,0,.6);border-radius:6px;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#9995a8}
        @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
        .modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:200;backdrop-filter:blur(6px)}
        .modal-panel{background:#0e0c16;border:1px solid rgba(255,255,255,.1);border-radius:16px;width:min(760px,95vw);max-height:85vh;display:flex;flex-direction:column;overflow:hidden}
        .th{border-radius:8px;overflow:hidden;cursor:pointer;aspect-ratio:1;border:2px solid transparent;transition:border-color .15s;background:rgba(255,255,255,.06);animation:pulse 1.5s ease infinite}
        .th:hover{border-color:#00d4ff}
        .th img{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .2s}
        .mi{flex:1;padding:6px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:8px;color:#ccc;font-size:12px;outline:none;font-family:inherit}
        .mi:focus{border-color:rgba(0,212,255,.4)}
        .sk{border-radius:8px;aspect-ratio:1;background:rgba(255,255,255,.04);animation:pulse 1.5s ease infinite}
      `}</style>
      <div style={{maxWidth:880,margin:"0 auto"}}>
        <div style={{marginBottom:24}}>
          <div className="dt">Djupuppskattning — Heuristik</div>
          <p style={{color:"#6b6580",fontSize:13,marginTop:6}}>Uppskatta djup via luminans, mättnad, kantskärpa & position</p>
        </div>
        <div className="gp" style={{padding:16,marginBottom:16}} onDragStart={e=>e.preventDefault()} onDragOver={e=>{e.preventDefault();e.stopPropagation()}} onDrop={e=>{e.preventDefault();e.stopPropagation()}}>
          <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"}}>
            <span style={{fontSize:11,color:"#6b6580",textTransform:"uppercase",letterSpacing:1,marginRight:4}}>Metod:</span>
            {Object.entries(METHODS).map(([k,v])=><button key={k} className={`mb ${method===k?"a":""}`} onClick={()=>{setMethod(k);if(image)processImage(image,k,blur)}} title={v.desc}>{v.label}</button>)}
            <div style={{flex:1}}/>
            {image&&<button className={`mb ${showSplit?"a":""}`} onClick={()=>setShowSplit(!showSplit)} style={{fontSize:11}}>{showSplit?"⊞ Sida vid sida":"⊟ Jämför"}</button>}
            {image&&rawDepthRef.current&&<button className="mb" onClick={()=>setShow3D(true)} style={{fontSize:11}}>◈ 3D</button>}
          </div>
          {image&&<div style={{display:"flex",gap:24,marginTop:14,alignItems:"center"}}>
            <label style={{fontSize:11,color:"#6b6580",display:"flex",alignItems:"center",gap:10,flex:1}} onMouseDown={e=>e.stopPropagation()}>
              Utjämning: <span style={{color:"#00d4ff",minWidth:20,textAlign:"center"}}>{blur}</span>
              <input type="range" min="0" max="8" value={blur} className="st" draggable={false} onChange={e=>{const v=+e.target.value;setBlur(v);handleBlurChange(v)}} style={{flex:1}}/>
            </label>
          </div>}
        </div>

        {!image?(
          <div className={`dz ${dragOver?"dg":""}`} onDrop={e=>{handleDrop(e);setDragOver(false)}} onDragOver={e=>{e.preventDefault();setDragOver(true)}} onDragLeave={()=>setDragOver(false)} onClick={()=>document.getElementById("hf-gallery").click()}>
            <input id="hf-gallery" type="file" accept="image/*" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
            <input id="hf-camera" type="file" accept="image/*" capture="environment" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" style={{marginBottom:16,opacity:dragOver?.9:.35,transition:"opacity .3s"}}><rect x="8" y="12" width="48" height="40" rx="6" stroke="#7b61ff" strokeWidth="2" fill="none"/><circle cx="22" cy="28" r="5" stroke="#00d4ff" strokeWidth="1.5" fill="none"/><path d="M8 42 L22 32 L34 40 L44 30 L56 42" stroke="#ff6ec7" strokeWidth="1.5" fill="none"/><path d="M32 4 L32 18 M26 12 L32 6 L38 12" stroke="#00d4ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            <div style={{fontSize:15,color:dragOver?"#00d4ff":"#8b84a0",marginBottom:6,fontFamily:"'Space Grotesk',sans-serif",fontWeight:600}}>{dragOver?"Släpp bilden här!":"Välj en bild"}</div>
            <div style={{fontSize:12,color:"#5a5470",marginBottom:12}}>Dra hit eller klicka för att öppna galleri</div>
            <div style={{fontSize:11,color:"#3d3855",display:"flex",gap:8,justifyContent:"center"}}><span style={{padding:"3px 8px",background:"rgba(255,255,255,.04)",borderRadius:4}}>PNG</span><span style={{padding:"3px 8px",background:"rgba(255,255,255,.04)",borderRadius:4}}>JPG</span><span style={{padding:"3px 8px",background:"rgba(255,255,255,.04)",borderRadius:4}}>WebP</span></div>
            <div style={{display:"flex",gap:8,marginTop:12,justifyContent:"center",flexWrap:"wrap"}}>
              <button className="mb" onClick={e=>{e.stopPropagation();document.getElementById("hf-camera").click();}}>
                ◎ Kamera
              </button>
              <button className="mb" onClick={e=>{e.stopPropagation();openModal();}}>
                ◈ Pexels
              </button>
            </div>
            <div className="gp" style={{padding:16,marginTop:24,textAlign:"left"}}>
              <div style={{fontSize:12,color:"#6b6580",lineHeight:1.9}}>
                <div><strong style={{color:"#9995a8"}}>Luminans</strong> — ljusare = närmare</div>
                <div><strong style={{color:"#9995a8"}}>Mättnad</strong> — starkare färger = närmare (atmosfärisk perspektiv)</div>
                <div><strong style={{color:"#9995a8"}}>Kontrast</strong> — skarpare lokala kanter = närmare</div>
                <div><strong style={{color:"#9995a8"}}>Kombinerad</strong> — kantskärpa (Sobel) + vertikal position + center-bias + luminans + mättnad + kontrast</div>
              </div>
            </div>
          </div>
        ):(
          <div>
            <div style={{display:"flex",gap:12,alignItems:"flex-start"}}>
              <div style={{flex:1,minWidth:0}}>
                {showSplit?(
                  <div className="gp" style={{padding:4,display:"flex",justifyContent:"center"}}>
                    <div ref={splitContainerRef} style={{position:"relative",userSelect:"none",cursor:"col-resize",display:"inline-block"}}
                      onMouseDown={e=>{dragging.current=true;handleSplitMove(e)}}
                      onTouchStart={e=>{dragging.current=true;handleSplitMove(e)}} onTouchMove={handleSplitMove} onTouchEnd={()=>{dragging.current=false}}>
                      <canvas ref={splitRef} className="sc"/>
                      <div style={{position:"absolute",top:0,bottom:0,left:`${splitX}%`,width:2,marginLeft:-1,background:"#fff",boxShadow:"0 0 8px rgba(0,0,0,.6)",pointerEvents:"none",zIndex:2}}/>
                      <div style={{position:"absolute",top:"50%",left:`${splitX}%`,transform:"translate(-50%,-50%)",width:32,height:32,borderRadius:"50%",background:"rgba(255,255,255,.95)",boxShadow:"0 2px 12px rgba(0,0,0,.4)",display:"flex",alignItems:"center",justifyContent:"center",pointerEvents:"none",zIndex:3}}>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M5 3L1 8L5 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M11 3L15 8L11 13" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </div>
                      <div className="cl" style={{left:8,zIndex:4}}>ORIGINAL</div>
                      <div className="cl" style={{right:8,left:"auto",zIndex:4}}>DJUP</div>
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
              {depthStats&&(
                <div className="gp" style={{padding:12,display:"flex",flexDirection:"column",gap:8,minWidth:160}}>
                  <span className="sp">Upplösning <span className="sv">{depthStats.resolution}</span></span>
                  <span className="sp">Medeldjup <span className="sv">{depthStats.avgDepth}%</span></span>
                  <span className="sp">Förgrund <span className="sv">{depthStats.fgPercent}%</span></span>
                  <div style={{flex:1}}/>
                  <button className="mb" onClick={()=>{setImage(null);setDepthStats(null);setShowSplit(false);rawDepthRef.current=null}} style={{fontSize:11,marginTop:8}}>✕ Ny bild</button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      <div style={{position:"absolute",left:-9999,top:-9999}}><canvas ref={origRef}/><canvas ref={depthRef}/></div>

      {showModal&&(
        <div className="modal-ov" onClick={()=>setShowModal(false)}>
          <div className="modal-panel" onClick={e=>e.stopPropagation()}>
            <div style={{padding:"16px 20px",borderBottom:"1px solid rgba(255,255,255,.08)",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
              <span style={{fontSize:14,fontWeight:600,color:"#ccc",fontFamily:"'Space Grotesk',sans-serif"}}>Välj ett foto</span>
              <button className="mb" style={{padding:"4px 10px"}} onClick={()=>setShowModal(false)}>✕</button>
            </div>
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
            <div style={{padding:16,overflowY:"auto",flex:1}}>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
                {modalLoading&&modalImages.length===0
                  ? Array.from({length:16}).map((_,i)=><div key={i} className="sk" style={{animationDelay:`${i*0.04}s`}}/>)
                  : modalImages.length===0
                    ? <div style={{gridColumn:"1/-1",textAlign:"center",padding:40,color:"#5a5470",fontSize:12}}>Inga bilder hittades</div>
                    : modalImages.map((img,i)=>(
                        <div key={i} className="th" onClick={()=>pickImage(img)} title={img.title}>
                          <img src={img.thumb} alt={img.title} loading="lazy" crossOrigin="anonymous" onLoad={e=>{e.currentTarget.style.opacity='1';e.currentTarget.parentElement.style.animation='none';}} onError={e=>{e.currentTarget.style.opacity='.25';e.currentTarget.parentElement.style.animation='none';}}/>
                        </div>
                      ))
                }
              </div>
              <div ref={sentinelRef} style={{height:40,display:"flex",alignItems:"center",justifyContent:"center"}}>
                {modalLoading&&modalImages.length>0&&<div style={{fontSize:11,color:"#4a4560",animation:"pulse 1.5s ease infinite"}}>Laddar fler…</div>}
              </div>
            </div>
            <div style={{padding:"10px 20px",borderTop:"1px solid rgba(255,255,255,.06)",fontSize:10,color:"#3a3550",flexShrink:0}}>
              Pexels · Gratis att använda · Foto av {modalImages[0]?.photographer ?? "…"} m.fl.
            </div>
          </div>
        </div>
      )}

      {show3D&&rawDepthRef.current&&(
        <Depth3DView
          depthData={rawDepthRef.current}
          depthW={dimRef.current.w}
          depthH={dimRef.current.h}
          imageCanvas={origRef.current}
          onClose={()=>setShow3D(false)}
        />
      )}
    </div>
  );
}
