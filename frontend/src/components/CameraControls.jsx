import React from 'react'
import './CameraControls.css'

const CameraControls = ({
  devices,
  selectedDeviceId,
  isStreaming,
  error,
  onDeviceChange,
  onStartCamera,
  onStopCamera,
  onFlipToggle,
  flipImage,
}) => {
  return (
    <div className="controls-below-video">
      <div className="camera-select-wrapper">
        <label htmlFor="camera-select" className="camera-select-label">
          Select Camera:
        </label>
        <select
          id="camera-select"
          value={selectedDeviceId}
          onChange={onDeviceChange}
          className="camera-select"
          disabled={!devices.length}
        >
          {devices.length === 0 ? (
            <option value="">No cameras available</option>
          ) : (
            devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
              </option>
            ))
          )}
        </select>
      </div>
      {isStreaming && (
        <button
          onClick={onStopCamera}
          className="control-button stop-button"
        >
          Stop Camera
        </button>
      )}
      {!isStreaming && !error && (
        <button
          onClick={onStartCamera}
          className="control-button start-button"
        >
          Start Camera
        </button>
      )}
      {isStreaming && (
        <button
          onClick={onFlipToggle}
          className={`control-button ${flipImage ? 'flip-active' : ''}`}
          title="Flip image horizontally"
        >
          {flipImage ? '🔄 Flip: ON' : '🔄 Flip: OFF'}
        </button>
      )}
    </div>
  )
}

export default CameraControls

