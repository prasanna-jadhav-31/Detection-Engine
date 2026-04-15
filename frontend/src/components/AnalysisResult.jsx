export function AnalysisResult({ result }) {
  if (!result) {
    return (
      <div className="panel result-panel empty-state">
        <p className="eyebrow">Result</p>
        <h2>Waiting for analysis</h2>
        <p className="muted-copy">
          Classification results will appear here after the backend finishes processing the image.
        </p>
      </div>
    );
  }

  const badgeClassName =
    result.prediction.toLowerCase() === 'ai' ? 'result-badge badge-ai' : 'result-badge badge-real';

  return (
    <div className="panel result-panel">
      <p className="eyebrow">Result</p>
      <h2>{result.prediction}</h2>
      <span className={badgeClassName}>{result.prediction}</span>
      <div className="metric-block">
        <span className="metric-label">Confidence</span>
        <span className="metric-value">{result.confidence}%</span>
      </div>
    </div>
  );
}

