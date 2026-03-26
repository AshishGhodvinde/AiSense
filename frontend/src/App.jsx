import { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import './index.css';

gsap.registerPlugin(ScrollTrigger);

// ── Signal card ───────────────────────────────────────────────────────────────
function SignalCard({ icon, label, status, text, index }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    gsap.fromTo(ref.current,
      { opacity: 0, x: 30 },
      { opacity: 1, x: 0, duration: 0.5, delay: 0.5 + index * 0.12, ease: 'power3.out' }
    );
  }, []);
  return (
    <div ref={ref} className={`signal-card signal-${status}`} style={{ opacity: 0 }}>
      <div className="signal-header">
        <span className="signal-icon">{icon}</span>
        <span className="signal-label">{label}</span>
        <span className={`signal-pill signal-pill-${status}`}>
          {status === 'ai' ? 'AI Sign' : status === 'real' ? 'Real Sign' : 'Neutral'}
        </span>
      </div>
      <p className="signal-text">{text}</p>
    </div>
  );
}

// ── Animated confidence bar ───────────────────────────────────────────────────
function ConfidenceBar({ value, isAi }) {
  const fillRef = useRef(null);
  const numRef = useRef(null);
  useEffect(() => {
    if (!fillRef.current) return;
    gsap.fromTo(fillRef.current, { width: '0%' },
      { width: `${value * 100}%`, duration: 1.3, ease: 'power2.out', delay: 0.25 });
    gsap.to({ v: 0 }, {
      v: value * 100, duration: 1.3, ease: 'power2.out', delay: 0.25,
      onUpdate() {
        if (numRef.current) numRef.current.textContent = this.targets()[0].v.toFixed(1) + '%';
      }
    });
  }, [value]);
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-labels">
        <span>Confidence</span>
        <span ref={numRef} className={`conf-num ${isAi ? 'text-ai' : 'text-real'}`}>0%</span>
      </div>
      <div className="conf-bar-bg">
        <div ref={fillRef} className={`conf-bar-fill ${isAi ? 'fill-ai' : 'fill-real'}`} style={{ width: '0%' }} />
      </div>
    </div>
  );
}

// ── Technical metrics row ─────────────────────────────────────────────────────
function MetricRow({ label, value, unit = '', hint = '' }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{typeof value === 'number' ? value.toFixed(2) : value}<span className="metric-unit">{unit}</span></span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [showEla, setShowEla] = useState(false);

  const vantaRef = useRef(null);
  const [vantaEffect, setVantaEffect] = useState(null);
  const imgColRef = useRef(null);
  const infoColRef = useRef(null);
  const badgeRef = useRef(null);
  const logoRef = useRef(null);

  // Header fade
  useEffect(() => {
    gsap.fromTo('.site-header',
      { opacity: 0, y: -20 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power3.out', delay: 0.1 });
  }, []);

  // Letter-by-letter logo reveal
  useEffect(() => {
    if (!logoRef.current) return;

    const text = 'AiSense';

    logoRef.current.innerHTML = text
      .split('')
      .map(ch => `<span class="logo-letter">${ch}</span>`)
      .join('');

    gsap.fromTo(
      '.logo-letter',
      {
        clipPath: 'inset(0 0 100% 0)',
        opacity: 0,
        y: 20
      },
      {
        clipPath: 'inset(0 0 0% 0)',
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: 'power4.out',
        stagger: 0.06,
        delay: 0.2
      }
    );
  }, []);

  // Vanta
  useEffect(() => {
    if (!vantaEffect && window.VANTA) {
      setVantaEffect(window.VANTA.WAVES({
        el: vantaRef.current,
        mouseControls: true, touchControls: true, gyroControls: false,
        minHeight: 200, minWidth: 200, scale: 1, scaleMobile: 1,
        color: 0x040412, shininess: 55, waveHeight: 20, waveSpeed: 0.85, zoom: 0.78,
      }));
    }
    return () => { if (vantaEffect) vantaEffect.destroy(); };
  }, [vantaEffect]);

  // Slide animation when result appears
  useLayoutEffect(() => {
    if (!result || !imgColRef.current || !infoColRef.current) return;
    const tl = gsap.timeline();
    // image column: start at center-ish, slide to final left position
    tl.fromTo(imgColRef.current,
      { x: '40%', opacity: 0 },
      { x: '0%', opacity: 1, duration: 0.75, ease: 'power3.out' }
    );
    // info column: starts from right off-screen, slides in
    tl.fromTo(infoColRef.current,
      { x: '60px', opacity: 0 },
      { x: '0px', opacity: 1, duration: 0.75, ease: 'power3.out' },
      '-=0.5'   // overlap 0.5s with the image slide
    );
    // badge pop
    if (badgeRef.current) {
      tl.fromTo(badgeRef.current,
        { scale: 0.5, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(2)' },
        '-=0.4'
      );
    }
  }, [result]);

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };
  const handleChange = (e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); };
  const handleFile = (f) => { setImage(URL.createObjectURL(f)); analyzeImage(f); };

  const analyzeImage = async (f) => {
    setLoading(true); setResult(null); setShowEla(false);
    const fd = new FormData(); fd.append('file', f);
    try {
      const res = await fetch('http://127.0.0.1:8000/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) { alert(data.error); resetState(); } else setResult(data);
    } catch { alert('Cannot reach backend. Make sure the FastAPI server is running.'); resetState(); }
    finally { setLoading(false); }
  };

  const resetState = () => {
    setImage(null); setResult(null); setShowEla(false);
  };

  const isAi = result?.prediction === 'AI Generated';
  const expl = result?.explanation || {};
  const hints = result?.heuristics || {};

  return (
    <div className="vanta-bg" ref={vantaRef}>
      <div className="app-wrap">

        {/* Header */}
        <header className="site-header">
          <div className="logo-wrap">
            <span className="logo-eye">👁</span>
            <h1 className="logo-text" ref={logoRef}></h1>
          </div>
          <p className="logo-sub">AI vs Real Image Detector</p>
        </header>

        {/* ── Upload ── */}
        {!image && !loading && (
          <div className="upload-glass">
            <div
              className={`drop-zone ${isDragActive ? 'drop-active' : ''}`}
              onDragEnter={handleDrag} onDragLeave={handleDrag}
              onDragOver={handleDrag} onDrop={handleDrop}
              onClick={() => document.getElementById('fu').click()}
            >
              <div className="drop-icon">🖼️</div>
              <div className="drop-title">Drop an image here</div>
              <div className="drop-hint">or click to browse — JPG, PNG, WEBP</div>
              <input id="fu" type="file" accept="image/*" onChange={handleChange} style={{ display: 'none' }} />
            </div>
          </div>
        )}

        {/* ── Loader ── */}
        {loading && (
          <div className="upload-glass">
            <div className="loader-wrap">
              <div className="scanner-ring"><div className="scanner-inner" /></div>
              <p className="loader-title">Analysing Image…</p>
              <p className="loader-sub">Running AI detection engine + signal analysis</p>
            </div>
          </div>
        )}

        {/* ── Results — full width split ── */}
        {result && !loading && (
          <div className="results-page">

            {/* LEFT: Image */}
            <div className="result-img-col" ref={imgColRef} style={{ opacity: 0 }}>
              <div className="img-frame">
                <img
                  src={showEla && result.ela_image ? `data:image/jpeg;base64,${result.ela_image}` : image}
                  alt="Analysed"
                />
                <div className={`img-overlay-tag ${isAi ? 'tag-ai' : 'tag-real'}`}>
                  {isAi ? '⚠ AI Generated' : '✓ Real Photo'}
                </div>
                {result.ela_image && (
                  <button className="ela-toggle" onClick={() => setShowEla(!showEla)}>
                    {showEla ? '📷 Original' : '🔬 ELA Heatmap'}
                  </button>
                )}
              </div>

              {/* Technical metrics under image */}
              <div className="tech-card">
                <h4 className="tech-title">⚙️ Technical Metrics</h4>
                <MetricRow label="ELA Variance" value={hints.ela_variance} hint={hints.ela_variance > 200 ? '⚠ High' : hints.ela_variance > 100 ? '~ Moderate' : '✓ Low'} />
                <MetricRow label="Noise Score" value={hints.noise_score} />
                <MetricRow label="Frequency AI Prob." value={hints.frequency_ai_prob} unit="%" hint={hints.frequency_ai_prob > 0.6 ? '⚠ Anomalies' : '✓ Normal'} />
                <MetricRow label="Noise AI Prob." value={hints.noise_ai_prob} unit="%" />
                <MetricRow label="Artifact AI Prob." value={hints.artifact_ai_prob} unit="%" />
              </div>

              <button className="btn-reset" onClick={resetState}>↩ Analyse Another Image</button>
            </div>

            {/* RIGHT: Info */}
            <div className="result-info-col" ref={infoColRef} style={{ opacity: 0 }}>

              {/* Badge */}
              <div ref={badgeRef} className={`verdict-badge ${isAi ? 'badge-ai' : 'badge-real'}`} style={{ opacity: 0 }}>
                <span className="badge-icon">{isAi ? '🤖' : '🟢'}</span>
                {result.prediction}
              </div>

              {/* Confidence */}
              <ConfidenceBar value={result.confidence} isAi={isAi} />

              {/* Verdict plain English */}
              {expl.headline && (
                <div className="verdict-headline-box">
                  <p className="verdict-headline">{expl.headline}</p>
                  <p className="verdict-sub">{expl.subline}</p>
                </div>
              )}

              {/* Camera / EXIF info — always shown prominently */}
              {hints.exif_reasons && hints.exif_reasons.length > 0 && (
                <div className="exif-card">
                  <h4 className="exif-title">📸 Camera & File Information</h4>
                  <ul className="exif-list">
                    {hints.exif_reasons.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              )}

              {/* Signal cards */}
              {expl.signals && expl.signals.length > 0 && (
                <div className="signals-section">
                  <h3 className="signals-title">🔎 What We Found</h3>
                  <div className="signals-list">
                    {expl.signals.map((s, i) => <SignalCard key={i} index={i} {...s} />)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <footer className="site-footer">Powered by Sightengine · HuggingFace ViT · ELA · FFT Analysis</footer>
      </div>
    </div>
  );
}

export default App;
