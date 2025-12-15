import React, { useRef, useEffect, useState } from 'react'
import DetectionTerminal from './DetectionTerminal'
import './CameraFeed.css'

const CameraFeed = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [detections, setDetections] = useState([])
  const [textDetections, setTextDetections] = useState([])
  const [currentFrameDetections, setCurrentFrameDetections] = useState([])
  const [currentFrameTextDetections, setCurrentFrameTextDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const detectionIntervalRef = useRef(null)
  
  const API_URL = 'http://localhost:8000'

  // Get available video input devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        // First, request permission to access devices
        await navigator.mediaDevices.getUserMedia({ video: true })
        
        // Then enumerate devices
        const deviceList = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput')
        
        setDevices(videoDevices)
        
        // Set the first device as default if available
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId)
        }
      } catch (err) {
        console.error('Error enumerating devices:', err)
      }
    }

    getDevices()
  }, [])

  // Function to capture frame and send to backend
  const captureAndDetect = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert canvas to base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      setIsProcessing(true)

      // Send to backend API
      const response = await fetch(`${API_URL}/detect-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      })

      if (!response.ok) {
        throw new Error('Detection request failed')
      }

      const data = await response.json()

      // Update object detections state for terminal
      if (data.detections && data.detections.length > 0) {
        setDetections(prev => {
          // Keep last 50 detections
          const newDetections = [...prev, ...data.detections]
          return newDetections.slice(-50)
        })
        // Store current frame detections for drawing boxes
        setCurrentFrameDetections(data.detections)
      } else {
        setCurrentFrameDetections([])
      }

      // Update text detections state for terminal
      if (data.text_detections && data.text_detections.length > 0) {
        setTextDetections(prev => {
          // Keep last 50 text detections
          const newTextDetections = [...prev, ...data.text_detections]
          return newTextDetections.slice(-50)
        })
        // Store current frame text detections for drawing boxes
        setCurrentFrameTextDetections(data.text_detections)
      } else {
        setCurrentFrameTextDetections([])
      }
    } catch (err) {
      console.error('Error detecting objects:', err)
    } finally {
      setIsProcessing(false)
    }
  }

  useEffect(() => {
    let stream = null

    const startCamera = async () => {
      if (!selectedDeviceId) return

      try {
        // Stop existing stream if any
        if (videoRef.current && videoRef.current.srcObject) {
          const existingStream = videoRef.current.srcObject
          existingStream.getTracks().forEach(track => track.stop())
        }

        // Clear any existing detection interval
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current)
          detectionIntervalRef.current = null
        }

        // Request camera access with specific device
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedDeviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsStreaming(true)
          setError(null)

          // Start detection loop (every 500ms)
          detectionIntervalRef.current = setInterval(() => {
            captureAndDetect()
          }, 500)
        }
      } catch (err) {
        console.error('Error accessing camera:', err)
        setError('Unable to access camera. Please ensure you have granted camera permissions.')
        setIsStreaming(false)
      }
    }

    if (selectedDeviceId) {
      startCamera()
    }

    // Cleanup function to stop the stream when component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
        detectionIntervalRef.current = null
      }
    }
  }, [selectedDeviceId])

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
    
    // Clear detection interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }

    // Clear detections and boxes
    setCurrentFrameDetections([])
    setDetections([])
    setCurrentFrameTextDetections([])
    setTextDetections([])
    
    // Clear overlay canvas
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height)
    }
  }

  const startCamera = async () => {
    if (!selectedDeviceId) {
      setError('Please select a camera device.')
      return
    }

    try {
      // Stop existing stream if any
      if (videoRef.current && videoRef.current.srcObject) {
        const existingStream = videoRef.current.srcObject
        existingStream.getTracks().forEach(track => track.stop())
      }

      // Clear any existing detection interval
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
        detectionIntervalRef.current = null
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedDeviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setError(null)

        // Start detection loop (every 500ms)
        detectionIntervalRef.current = setInterval(() => {
          captureAndDetect()
        }, 500)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Unable to access camera. Please ensure you have granted camera permissions.')
      setIsStreaming(false)
    }
  }

  const handleDeviceChange = (event) => {
    setSelectedDeviceId(event.target.value)
  }

  // Draw bounding boxes on overlay canvas
  useEffect(() => {
    const drawBoxes = () => {
      if (!overlayCanvasRef.current || !videoRef.current) {
        // Clear canvas if no video
        if (overlayCanvasRef.current) {
          const ctx = overlayCanvasRef.current.getContext('2d')
          ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height)
        }
        return
      }

      const video = videoRef.current
      const overlayCanvas = overlayCanvasRef.current
      const ctx = overlayCanvas.getContext('2d')

      // Get video display dimensions
      const videoRect = video.getBoundingClientRect()
      const videoDisplayWidth = videoRect.width
      const videoDisplayHeight = videoRect.height
      const videoActualWidth = video.videoWidth
      const videoActualHeight = video.videoHeight

      if (videoActualWidth === 0 || videoActualHeight === 0) return

      // Set overlay canvas size to match video display size
      overlayCanvas.width = videoDisplayWidth
      overlayCanvas.height = videoDisplayHeight

      // Calculate scale factors
      const scaleX = videoDisplayWidth / videoActualWidth
      const scaleY = videoDisplayHeight / videoActualHeight

      // Clear previous drawings
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)

      // Draw object detection bounding boxes (accounting for video mirroring)
      currentFrameDetections.forEach((detection) => {
        const { bbox, class: className, confidence } = detection

        // Scale bounding box coordinates to display size
        // Mirror horizontally: x = width - x (because video is mirrored)
        const x1 = videoDisplayWidth - (bbox.x2 * scaleX)
        const y1 = bbox.y1 * scaleY
        const x2 = videoDisplayWidth - (bbox.x1 * scaleX)
        const y2 = bbox.y2 * scaleY
        const width = x2 - x1
        const height = y2 - y1

        // Draw bounding box (green for objects)
        ctx.strokeStyle = '#00ff00'
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, width, height)

        // Draw label background
        const labelText = `${className} ${confidence}%`
        ctx.font = 'bold 16px Arial'
        const textMetrics = ctx.measureText(labelText)
        const labelWidth = textMetrics.width + 10
        const labelHeight = 24

        // Position label at top-left of box
        const labelX = Math.min(x1, x2)
        const labelY = Math.max(0, Math.min(y1, y2) - labelHeight)

        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)'
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight)

        // Draw label text
        ctx.fillStyle = '#000000'
        ctx.fillText(labelText, labelX + 5, labelY + 18)
      })

      // Draw text detection bounding boxes (accounting for video mirroring)
      currentFrameTextDetections.forEach((detection) => {
        const { bbox, text, confidence } = detection

        // Scale bounding box coordinates to display size
        // Mirror horizontally: x = width - x (because video is mirrored)
        const x1 = videoDisplayWidth - (bbox.x2 * scaleX)
        const y1 = bbox.y1 * scaleY
        const x2 = videoDisplayWidth - (bbox.x1 * scaleX)
        const y2 = bbox.y2 * scaleY
        const width = x2 - x1
        const height = y2 - y1

        // Draw bounding box (blue for text)
        ctx.strokeStyle = '#0080ff'
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, width, height)

        // Draw label background
        const labelText = `"${text}" ${confidence}%`
        ctx.font = 'bold 16px Arial'
        const textMetrics = ctx.measureText(labelText)
        const labelWidth = textMetrics.width + 10
        const labelHeight = 24

        // Position label at top-left of box
        const labelX = Math.min(x1, x2)
        const labelY = Math.max(0, Math.min(y1, y2) - labelHeight)

        ctx.fillStyle = 'rgba(0, 128, 255, 0.8)'
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight)

        // Draw label text
        ctx.fillStyle = '#ffffff'
        ctx.fillText(labelText, labelX + 5, labelY + 18)
      })
    }

    drawBoxes()

    // Redraw on window resize
    const handleResize = () => {
      drawBoxes()
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [currentFrameDetections, currentFrameTextDetections])

  return (
    <div className="camera-container">
      <div className="video-wrapper">
        {error ? (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={startCamera} className="retry-button">
              Retry
            </button>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />
            <canvas 
              ref={overlayCanvasRef} 
              className="overlay-canvas"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </>
        )}
      </div>
      
      <div className="controls-panel">
        <div className="controls">
          <div className="camera-select-wrapper">
            <label htmlFor="camera-select" className="camera-select-label">
              Select Camera:
            </label>
            <select
              id="camera-select"
              value={selectedDeviceId}
              onChange={handleDeviceChange}
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
              onClick={stopCamera}
              className="control-button stop-button"
            >
              Stop Camera
            </button>
          )}
          {!isStreaming && !error && (
            <button
              onClick={startCamera}
              className="control-button start-button"
            >
              Start Camera
            </button>
          )}
        </div>

        {isStreaming && (
          <div className="status-indicator">
            <span className="status-dot"></span>
            Camera Active
          </div>
        )}

        <DetectionTerminal 
          detections={detections}
          textDetections={textDetections}
          isProcessing={isProcessing}
        />
      </div>
    </div>
  )
}

export default CameraFeed

