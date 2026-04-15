import { useRef } from 'react';


export function FileUploader({
  error,
  isSubmitting,
  previewUrl,
  onAnalyze,
  onFileSelect,
  onReset,
}) {
  const inputRef = useRef(null);

  const handleFiles = (nextFile) => {
    if (nextFile && nextFile.type.startsWith('image/')) {
      onFileSelect(nextFile);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    handleFiles(event.dataTransfer?.files?.[0]);
  };

  const openPicker = () => {
    inputRef.current?.click();
  };

  return (
    <div className="panel">
      {!previewUrl ? (
        <button
          className="upload-dropzone"
          type="button"
          onClick={openPicker}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
        >
          <span className="upload-title">Select or drop an image</span>
          <span className="upload-copy">Supports JPEG, PNG, and WEBP.</span>
        </button>
      ) : (
        <div className="preview-stack">
          <img className="preview-image" src={previewUrl} alt="Selected upload preview" />
          <div className="actions-row">
            <button className="primary-button" type="button" disabled={isSubmitting} onClick={onAnalyze}>
              {isSubmitting ? 'Analyzing...' : 'Analyze'}
            </button>
            <button className="secondary-button" type="button" disabled={isSubmitting} onClick={onReset}>
              Reset
            </button>
          </div>
        </div>
      )}

      <input
        ref={inputRef}
        hidden
        accept="image/jpeg,image/png,image/webp"
        type="file"
        onChange={(event) => handleFiles(event.target.files?.[0])}
      />

      {error ? <p className="error-text">{error}</p> : null}
    </div>
  );
}

