import React, { useState, useEffect, useRef } from 'react'
import { useCamera } from '../hooks/useCamera'
import { useDetection } from '../hooks/useDetection'
import { useCanvasDrawing } from '../hooks/useCanvasDrawing'
import CameraControls from './CameraControls'
import VideoDisplay from './VideoDisplay'
import DetectionTerminal from './DetectionTerminal'
import { CAMERA_CONFIG } from '../constants/config'
import './CameraFeed.css'

const CameraFeed = () => {
  const canvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const [flipImage, setFlipImage] = useState(false)
  const [requestInterval, setRequestInterval] = useState(CAMERA_CONFIG.DEFAULT_REQUEST_INTERVAL)

  // Use custom hooks
  const {
    devices,
    selectedDeviceId,
    isStreaming,
    error,
    videoRef,
    startCamera,
    stopCamera,
    handleDeviceChange,
  } = useCamera()

  const {
    detections,
    currentFrameDetections,
    ocrTextDetections,
    currentFrameOcrDetections,
    isProcessing,
    backendLogs,
    captureAndDetect,
    startDetectionInterval,
    stopDetectionInterval,
    clearDetections,
  } = useDetection(videoRef, canvasRef, flipImage, requestInterval)

  // Use canvas drawing hook
  useCanvasDrawing(
    videoRef,
    overlayCanvasRef,
    currentFrameDetections,
    currentFrameOcrDetections,
    flipImage
  )

  // Start/stop detection interval based on streaming state
  useEffect(() => {
    if (isStreaming) {
      startDetectionInterval()
    } else {
      stopDetectionInterval()
    }

    return () => {
      stopDetectionInterval()
    }
  }, [isStreaming, startDetectionInterval, stopDetectionInterval])

  // Handle camera stop - clear detections
  const handleStopCamera = () => {
    stopCamera()
    stopDetectionInterval()
    clearDetections()

    // Clear overlay canvas
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height)
    }
  }

  const handleFlipToggle = () => {
    setFlipImage(!flipImage)
  }

  return (
    <div className="camera-container">
      <div className="left-section">
        <VideoDisplay
          videoRef={videoRef}
          overlayCanvasRef={overlayCanvasRef}
          canvasRef={canvasRef}
          error={error}
          flipImage={flipImage}
          onRetry={startCamera}
        />

        <CameraControls
          devices={devices}
          selectedDeviceId={selectedDeviceId}
          isStreaming={isStreaming}
          error={error}
          onDeviceChange={handleDeviceChange}
          onStartCamera={startCamera}
          onStopCamera={handleStopCamera}
          onFlipToggle={handleFlipToggle}
          flipImage={flipImage}
        />
      </div>

      <div className="terminal-panel">
        <DetectionTerminal
          detections={detections}
          ocrTextDetections={ocrTextDetections}
          isProcessing={isProcessing}
          backendLogs={backendLogs}
        />
      </div>
    </div>
  )
}

export default CameraFeed
