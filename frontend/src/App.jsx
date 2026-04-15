import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Image as ImageIcon, Sparkles, Shield, Cpu, ArrowLeft, RefreshCw, Layers } from 'lucide-react';

const App = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0] || e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileUpload(file);
    }
  }, []);

  const handleFileUpload = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setError(null);
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setShowHeatmap(false);
  };

  return (
    <div className="immersive-container">
      {/* Header */}
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 style={{ fontSize: '3.5rem', marginBottom: '10px' }}>
            Classifier <span className="gradient-text">Engine</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem' }}>
            Generative Synthesis Visualization & Diagnostics
          </p>
        </motion.div>
      </header>

      <AnimatePresence modeS="wait">
        {!result ? (
          <motion.div
            key="upload-view"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            className="glass-panel"
          >
            {!preview ? (
              <div
                className="drop-zone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                onClick={() => document.getElementById('file-input').click()}
                style={{
                  border: '2px dashed var(--glass-border)',
                  borderRadius: '16px',
                  padding: '80px 40px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  transition: 'background 0.3s'
                }}
              >
                <input
                  type="file"
                  id="file-input"
                  hidden
                  accept="image/jpeg,image/png,image/webp"
                  onChange={onDrop}
                />
                <Upload size={48} color="var(--primary)" style={{ marginBottom: '20px' }} />
                <h3>Drop your image here</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '10px' }}>
                  Supports JPEG, PNG, WEBP (Max 10MB)
                </p>
              </div>
            ) : (
              <div style={{ position: 'relative' }}>
                <img
                  src={preview}
                  alt="Preview"
                  style={{ width: '100%', borderRadius: '16px', maxHeight: '500px', objectFit: 'contain' }}
                />
                <div style={{ marginTop: '20px', display: 'flex', gap: '15px' }}>
                  <button
                    onClick={analyzeImage}
                    disabled={loading}
                    className="glass-panel"
                    style={{
                      flex: 1,
                      padding: '15px',
                      background: 'var(--primary)',
                      border: 'none',
                      color: 'white',
                      fontWeight: 'bold',
                      fontSize: '1.1rem',
                      cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {loading ? <RefreshCw className="animate-spin" /> : 'Check'}
                  </button>
                  <button
                    onClick={reset}
                    disabled={loading}
                    style={{ padding: '0 20px', background: 'transparent', border: '1px solid var(--glass-border)', color: 'white', borderRadius: '12px', cursor: 'pointer' }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {error && (
              <p style={{ color: '#ff4757', marginTop: '20px', textAlign: 'center' }}>
                {error}
              </p>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="result-view"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            className="immersive-results"
            style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 350px', gap: '30px' }}
          >
            {/* Visual Section */}
            <div className="glass-panel" style={{ overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <ImageIcon size={20} />
                  <span>Graphic Canvas</span>
                </div>
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  style={{
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center',
                    padding: '8px 16px',
                    background: showHeatmap ? 'var(--primary)' : 'rgba(255,255,255,0.1)',
                    border: 'none',
                    color: 'white',
                    borderRadius: '8px',
                    cursor: 'pointer'
                  }}
                >
                  <Layers size={16} /> {showHeatmap ? 'Original' : 'Heatmap'}
                </button>
              </div>

              <div style={{ position: 'relative', height: '450px', width: '100%', background: '#000', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <motion.img
                  layoutId="image-main"
                  src={showHeatmap ? result.heatmap_url : preview}
                  alt="Analysis"
                  style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                />
              </div>
            </div>

            {/* Sidebar Data */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="glass-panel" style={{ textAlign: 'center' }}>
                <Shield size={32} color="var(--accent)" style={{ marginBottom: '15px' }} />
                <h3>Classification</h3>
                <div style={{ margin: '20px 0' }}>
                  <span className={`badge badge-${result.prediction.toLowerCase()}`}>
                    {result.prediction}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <div className="confidence-gauge" style={{ '--percent': result.confidence }}>
                    <div className="confidence-text">{result.confidence}%</div>
                  </div>
                </div>
                <p style={{ color: 'var(--text-secondary)', marginTop: '10px', fontSize: '0.9rem' }}>
                  Confidence %
                </p>
              </div>

              <div className="glass-panel">
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '15px' }}>
                  <Cpu size={20} color="var(--primary)" />
                  <span>Diagnostics</span>
                </div>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'rgba(255,255,255,0.9)' }}>
                  {result.explanation}
                </p>
                <div style={{ marginTop: '20px' }}>
                  <button
                    onClick={reset}
                    style={{ width: '100%', padding: '12px', background: 'transparent', border: '1px solid var(--glass-border)', color: 'white', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', cursor: 'pointer' }}
                  >
                    <ArrowLeft size={16} /> New Analysis
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer style={{ marginTop: 'auto', padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        <p>© 2026 ML Core · Inage Classifier Engine</p>
      </footer>

      {/* Internal CSS for Gauge */}
      <style>{`
        .confidence-gauge {
          --percent: 0;
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default App;
