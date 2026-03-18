import { useState } from 'react';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [showEla, setShowEla] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    setFile(selectedFile);
    setImage(URL.createObjectURL(selectedFile));
    analyzeImage(selectedFile);
  };

  const analyzeImage = async (fileToAnalyze) => {
    setLoading(true);
    setResult(null);
    setShowEla(false);
    
    const formData = new FormData();
    formData.append('file', fileToAnalyze);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if(data.error) {
        alert(data.error);
        resetState();
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to backend or invalid response. Make sure the FastAPI server is running.');
      resetState();
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setImage(null);
    setFile(null);
    setResult(null);
    setShowEla(false);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>AiSense</h1>
        <p>Premium AI vs Real Image Detection</p>
      </header>

      <main className="glass-panel">
        {!image && !loading && (
          <div 
            className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <div className="upload-icon">📸</div>
            <div className="upload-text">Drag & Drop an image here</div>
            <div className="upload-hint">or click to browse your files</div>
            <input 
              id="file-upload" 
              type="file" 
              accept="image/*" 
              onChange={handleChange} 
              style={{ display: 'none' }} 
            />
          </div>
        )}

        {loading && (
          <div className="loader-container">
            <div className="spinner"></div>
            <h3>Analyzing Image...</h3>
            <p className="upload-hint">Running Machine Learning Inference & Heuristics</p>
          </div>
        )}

        {result && !loading && (
          <div className="results-grid">
            <div className="result-left">
              <div className="result-image-container">
                <img 
                  src={showEla && result.ela_image ? result.ela_image : image} 
                  alt="Uploaded preview" 
                />
                {result.ela_image && (
                  <button 
                    className="toggle-view-btn"
                    onClick={() => setShowEla(!showEla)}
                  >
                    {showEla ? "Original Image" : "Show ELA Heatmap"}
                  </button>
                )}
              </div>
            </div>

            <div className="result-right">
              <div className={`prediction-badge ${result.prediction === 'AI Generated' ? 'ai' : 'real'}`}>
                {result.prediction}
              </div>
              
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Confidence</span>
                  <span>{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-bg">
                  <div 
                    className="confidence-bar-fill" 
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <h3>Technical Metrics</h3>
              <div className="data-row">
                <span className="data-label">High-Frequency Noise Score</span>
                <span style={{fontWeight: '500'}}>{result.heuristics.noise_score}</span>
              </div>
              <div className="data-row">
                <span className="data-label">ELA Variance</span>
                <span style={{fontWeight: '500'}}>{result.heuristics.ela_variance}</span>
              </div>

              {result.heuristics.exif_reasons && result.heuristics.exif_reasons.length > 0 && (
                <div style={{ marginTop: '1.5rem', background: 'rgba(0,0,0,0.1)', padding: '1.2rem', borderRadius: '12px' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--accent)' }}>Metadata Indicators</h4>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', color: 'var(--text-secondary)' }}>
                    {result.heuristics.exif_reasons.map((r, i) => (
                      <li key={i} style={{ marginBottom: '0.3rem' }}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="explanation-box">
                <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--accent)' }}>Explanation</h4>
                {result.explanation}
              </div>

              <button className="reset-btn" onClick={resetState}>
                Analyze Another Image
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
