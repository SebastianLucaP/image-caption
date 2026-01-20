import React from 'react';

const CaptionDisplay = ({ caption }) => {
    if (!caption) return null;

    return (
        <div className="caption-container" aria-live="polite">
            <h2>Generated Caption:</h2>
            <p className="caption-text">{caption}</p>
        </div>
    );
};

export default CaptionDisplay;
