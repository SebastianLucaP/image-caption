import React, { useState } from 'react';
import axios from 'axios';
import ImageUploader from './components/ImageUploader';
import CaptionDisplay from './components/CaptionDisplay';
import TTSControl from './components/TTSControl';
import ModelSelector from './components/ModelSelector';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedModel, setSelectedModel] = useState('vi-gpt2');
  const [caption, setCaption] = useState('');
  const [audioBase64, setAudioBase64] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setCaption('');
    setAudioBase64('');
    setError('');
  };

  const handleGenerateCaption = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setCaption('');
    setAudioBase64('');
    setError('');

    const formData = new FormData();
    formData.append('image', selectedImage);

    // Determine model string for backend
    const modelMap = { 'vi-gpt2': 'nic', 'blip': 'blip', 'custom': 'custom' };
    const backendModel = modelMap[selectedModel] || 'nic';

    formData.append('model', backendModel);

    try {
      const response = await axios.post('http://localhost:5000/caption', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      if (response.data && response.data.caption) {
        setCaption(response.data.caption);
        if (response.data.audio) {
          setAudioBase64(response.data.audio);
        }
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      console.error(err);
      setError('Failed to generate caption. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Image Captioning</h1>
      </header>

      <main>
        <ModelSelector
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />

        <ImageUploader
          onImageSelect={handleImageSelect}
          selectedImage={selectedImage}
        />

        {selectedImage && (
          <button
            className="generate-button"
            onClick={handleGenerateCaption}
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? 'Generating...' : 'Generate Caption'}
          </button>
        )}

        {error && <p className="error-message" role="alert">{error}</p>}

        <ErrorBoundary>
          <CaptionDisplay caption={caption} />
          <TTSControl audioBase64={audioBase64} />
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
