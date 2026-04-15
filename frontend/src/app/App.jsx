import { useState } from 'react';

import { analyzeImage } from '../services/api';
import { FileUploader } from '../components/FileUploader';
import { AnalysisResult } from '../components/AnalysisResult';


export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const resetState = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setIsSubmitting(false);
  };

  const handleFileSelect = (nextFile) => {
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) {
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const analysisResult = await analyzeImage(file);
      setResult(analysisResult);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">Image Classifier</p>
        <h1>Detect AI-generated images.</h1>
        <p className="hero-copy">
          Upload a JPEG, PNG, or WEBP image to classify it as AI-generated or real.
        </p>
      </section>

      <section className="workspace-panel">
        <div className="workspace-main">
          <FileUploader
            error={error}
            isSubmitting={isSubmitting}
            previewUrl={previewUrl}
            onAnalyze={handleAnalyze}
            onFileSelect={handleFileSelect}
            onReset={resetState}
          />
        </div>

        <aside className="workspace-sidebar">
          <AnalysisResult result={result} />
        </aside>
      </section>
    </main>
  );
}

