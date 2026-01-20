import React, { useEffect, useRef } from 'react';

const TTSControl = ({ audioBase64 }) => {
    const audioRef = useRef(null);

    useEffect(() => {
        if (audioBase64 && audioRef.current) {
            audioRef.current.load();
            audioRef.current.play().catch(e => console.error("Auto-play blocked:", e));
        }
    }, [audioBase64]);

    const playAudio = () => {
        if (audioRef.current) {
            audioRef.current.currentTime = 0;
            audioRef.current.play().catch(e => console.error("Play failed:", e));
        }
    };

    if (!audioBase64) return null;

    return (
        <div className="tts-container">
            <audio ref={audioRef} style={{ display: 'none' }}>
                <source src={`data:audio/mp3;base64,${audioBase64}`} type="audio/mp3" />
            </audio>
            <button
                className="tts-button"
                onClick={playAudio}
                aria-label="Read caption again"
            >
                Read Again
            </button>
        </div>
    );
};

export default TTSControl;
