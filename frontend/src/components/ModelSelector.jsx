import React from 'react';

const ModelSelector = ({ selectedModel, onModelChange }) => {
    return (
        <div className="model-selector-container">
            <label htmlFor="model-select">Select Captioning Model:</label>
            <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => onModelChange(e.target.value)}
                className="model-select"
            >
                <option value="vi-gpt2">NIC (VIT-GPT2)</option>
                <option value="blip">BLIP (State of the Art)</option>
                <option value="custom">Custom (CNN+LSTM+Attention)</option>
            </select>
        </div>
    );
};

export default ModelSelector;
