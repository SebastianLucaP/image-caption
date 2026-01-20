import React, { useRef } from 'react';

const ImageUploader = ({ onImageSelect, selectedImage }) => {
    const fileInputRef = useRef(null);

    const handleClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            onImageSelect(file);
        }
    };

    return (
        <div className="image-uploader-container">
            <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                aria-label="Upload Image"
            />
            <button
                className="upload-button"
                onClick={handleClick}
                aria-label="Select an image to upload"
            >
                {selectedImage ? 'Change Image' : 'Select Image'}
            </button>

            {selectedImage && (
                <div className="image-preview">
                    <img src={URL.createObjectURL(selectedImage)} alt="Preview" />
                </div>
            )}
        </div>
    );
};

export default ImageUploader;
