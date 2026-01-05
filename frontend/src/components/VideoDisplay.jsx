import React from 'react'
import './VideoDisplay.css'

const VideoDisplay = ({
  videoRef,
  overlayCanvasRef,
  canvasRef,
  error,
  flipImage,
  onRetry,
}) => {
  if (error) {
    return (
      <div className="video-wrapper">
        <div className="error-message">
          <p>{error}</p>
          <button onClick={onRetry} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="video-wrapper">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-video"
        style={{
          transform: flipImage ? 'scaleX(1)' : 'scaleX(-1)',
        }}
      />
      <canvas
        ref={overlayCanvasRef}
        className="overlay-canvas"
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  )
}

export default VideoDisplay

