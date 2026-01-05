import React, { useRef, useEffect, useState, useCallback } from 'react'
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
  const [currentFrameDetections, setCurrentFrameDetections] = useState([])
  const [ocrTextDetections, setOcrTextDetections] = useState([])
  const [currentFrameOcrDetections, setCurrentFrameOcrDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [flipImage, setFlipImage] = useState(false)
  const [requestInterval, setRequestInterval] = useState(500) // milliseconds between requests
  const [backendLogs, setBackendLogs] = useState([])
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
  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return

    const video = videoRef.current
    
    // Check if video is ready and has valid dimensions
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      // Video not ready yet, skip this frame
      return
    }

    try {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw current video frame to canvas (with optional flip)
      if (flipImage) {
        // Flip horizontally by scaling and translating
        ctx.save()
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        ctx.restore()
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      }

      // Convert canvas to base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      setIsProcessing(true)

      // Send to backend API
      let response
      try {
        response = await fetch(`${API_URL}/input-image-base64`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        })
      } catch (fetchError) {
        console.error('Network error:', fetchError)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] Network error: ${fetchError.message}. Is the backend API server running on ${API_URL}?`]
          return newLogs.slice(-100)
        })
        throw fetchError
      }

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API error:', response.status, errorText)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] API request failed: ${response.status} ${response.statusText}`]
          return newLogs.slice(-100)
        })
        throw new Error(`Detection request failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      
      // Log any errors from backend
      if (data.error) {
        console.error('Backend error:', data.error)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] ${data.error}`]
          return newLogs.slice(-100)
        })
      }

      // Extract logs from backend
      if (data.logs && Array.isArray(data.logs)) {
        setBackendLogs(prev => {
          // Keep last 100 logs
          const newLogs = [...prev, ...data.logs]
          return newLogs.slice(-100)
        })
      }

      // Extract YOLO detections from the new API response structure
      const yoloDetections = data.yolo_result?.detections || []
      const ocrResults = data.paddleocr_result || null
      const ocrTextDetections = ocrResults?.text_detections || []

      // Update detections state for terminal
      if (yoloDetections.length > 0) {
        setDetections(prev => {
          // Keep last 50 detections
          const newDetections = [...prev, ...yoloDetections]
          return newDetections.slice(-50)
        })
        // Store current frame detections for drawing boxes
        setCurrentFrameDetections(yoloDetections)
      } else {
        setCurrentFrameDetections([])
      }

      // Update OCR text detections state for terminal and drawing
      if (ocrTextDetections.length > 0) {
        setOcrTextDetections(prev => {
          // Keep last 50 OCR detections
          const newOcrDetections = [...prev, ...ocrTextDetections]
          return newOcrDetections.slice(-50)
        })
        // Store current frame OCR detections for drawing boxes
        setCurrentFrameOcrDetections(ocrTextDetections)
      } else {
        setCurrentFrameOcrDetections([])
      }
    } catch (err) {
      console.error('Error detecting objects:', err)
    } finally {
      setIsProcessing(false)
    }
  }, [isProcessing, flipImage, API_URL])

  // Track if camera was explicitly stopped by user
  const cameraStoppedRef = useRef(false)
  const streamRef = useRef(null)

  // Handle video stream events
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      // Stream is ready
      setIsStreaming(true)
      setError(null)
    }

    const handleError = (e) => {
      console.error('Video element error:', e)
      setError('Video stream error. Trying to reconnect...')
      setIsStreaming(false)
      // Try to restart camera after a short delay
      setTimeout(() => {
        if (!cameraStoppedRef.current && selectedDeviceId) {
          // Trigger restart by updating a dependency
          setSelectedDeviceId(prev => prev) // This will trigger the camera useEffect
        }
      }, 1000)
    }

    const handleEnded = () => {
      console.warn('Video stream ended unexpectedly')
      setError('Camera stream ended. Trying to reconnect...')
      setIsStreaming(false)
      // Try to restart camera
      if (!cameraStoppedRef.current && selectedDeviceId) {
        setTimeout(() => {
          setSelectedDeviceId(prev => prev)
        }, 1000)
      }
    }

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('error', handleError)
    video.addEventListener('ended', handleEnded)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('error', handleError)
      video.removeEventListener('ended', handleEnded)
    }
  }, [selectedDeviceId])

  useEffect(() => {
    let stream = null
    let isMounted = true

    const startCamera = async () => {
      // Don't auto-start if user explicitly stopped the camera
      if (!selectedDeviceId || cameraStoppedRef.current || !isMounted) return

      try {
        // Stop existing stream if any
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => {
            track.stop()
            track.enabled = false
          })
          streamRef.current = null
        }

        if (videoRef.current && videoRef.current.srcObject) {
          const existingStream = videoRef.current.srcObject
          existingStream.getTracks().forEach(track => {
            track.stop()
            track.enabled = false
          })
          videoRef.current.srcObject = null
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

        if (!isMounted) {
          // Component unmounted, stop the stream
          stream.getTracks().forEach(track => track.stop())
          return
        }

        // Store stream reference
        streamRef.current = stream

        // Add event listeners to track stream state
        stream.getTracks().forEach(track => {
          track.onended = () => {
            console.warn('Camera track ended:', track.kind)
            if (!cameraStoppedRef.current && isMounted) {
              setError('Camera track ended. Trying to reconnect...')
              setIsStreaming(false)
              setTimeout(() => {
                if (isMounted && !cameraStoppedRef.current) {
                  setSelectedDeviceId(prev => prev)
                }
              }, 1000)
            }
          }

          track.onerror = (e) => {
            console.error('Camera track error:', e)
            if (!cameraStoppedRef.current && isMounted) {
              setError('Camera track error. Trying to reconnect...')
              setIsStreaming(false)
            }
          }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsStreaming(true)
          setError(null)
          cameraStoppedRef.current = false // Reset stopped flag when starting

          // Start detection loop with configurable interval
          detectionIntervalRef.current = setInterval(() => {
            if (isMounted && videoRef.current && videoRef.current.readyState >= 2) {
              captureAndDetect()
            }
          }, requestInterval)
        }
      } catch (err) {
        console.error('Error accessing camera:', err)
        if (isMounted) {
          setError(`Unable to access camera: ${err.message}. Please ensure you have granted camera permissions.`)
          setIsStreaming(false)
        }
      }
    }

    if (selectedDeviceId && !cameraStoppedRef.current) {
      startCamera()
    }

    // Cleanup function to stop the stream when component unmounts or dependencies change
    return () => {
      isMounted = false
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
        streamRef.current = null
      }
      if (stream) {
        stream.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
        detectionIntervalRef.current = null
      }
    }
  }, [selectedDeviceId, requestInterval]) // Removed captureAndDetect from dependencies

  // Update detection interval when requestInterval changes (without restarting camera)
  useEffect(() => {
    if (isStreaming && detectionIntervalRef.current) {
      // Clear existing interval
      clearInterval(detectionIntervalRef.current)
      // Start new interval with updated frequency
      detectionIntervalRef.current = setInterval(() => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          captureAndDetect()
        }
      }, requestInterval)
    }
  }, [requestInterval, isStreaming, captureAndDetect])

  const stopCamera = () => {
    // Mark camera as explicitly stopped
    cameraStoppedRef.current = true
    
    // Stop stream reference
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop()
        track.enabled = false
      })
      streamRef.current = null
    }
    
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject
      // Stop all tracks to release the camera
      stream.getTracks().forEach(track => {
        track.stop()
        track.enabled = false
      })
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
    setCurrentFrameOcrDetections([])
    setDetections([])
    setOcrTextDetections([])
    setBackendLogs([])
    
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

    // Reset stopped flag when user explicitly starts camera
    cameraStoppedRef.current = false

    try {
      // Stop existing stream if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
        streamRef.current = null
      }

      if (videoRef.current && videoRef.current.srcObject) {
        const existingStream = videoRef.current.srcObject
        existingStream.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
        videoRef.current.srcObject = null
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

      // Store stream reference
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setError(null)

        // Start detection loop with configurable interval
        detectionIntervalRef.current = setInterval(() => {
          if (videoRef.current && videoRef.current.readyState >= 2) {
            captureAndDetect()
          }
        }, requestInterval)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError(`Unable to access camera: ${err.message}. Please ensure you have granted camera permissions.`)
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

      // Draw YOLO bounding boxes (accounting for video mirroring/flipping)
      currentFrameDetections.forEach((detection) => {
        const { bbox, class: className, confidence } = detection

        // Scale bounding box coordinates to display size
        // If video is mirrored (flipImage = false), mirror the x coordinates
        // If video is not mirrored (flipImage = true), use coordinates as-is
        let x1, x2
        if (flipImage) {
          // Video is not mirrored, use coordinates directly
          x1 = bbox.x1 * scaleX
          x2 = bbox.x2 * scaleX
        } else {
          // Video is mirrored, flip x coordinates
          x1 = videoDisplayWidth - (bbox.x2 * scaleX)
          x2 = videoDisplayWidth - (bbox.x1 * scaleX)
        }
        const y1 = bbox.y1 * scaleY
        const y2 = bbox.y2 * scaleY
        const width = x2 - x1
        const height = y2 - y1

        // Draw bounding box (green for YOLO)
        ctx.strokeStyle = '#00ff00'
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, width, height)

        // Draw label background
        const labelText = `${className} ${confidence.toFixed(1)}%`
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

      // Draw OCR text boxes (polygon-based)
      currentFrameOcrDetections.forEach((ocrDetection) => {
        // OCR detection format: [text, confidence, polygon]
        const [text, confidence, polygon] = ocrDetection

        if (!polygon || polygon.length !== 4) return

        // Scale polygon coordinates to display size
        const scaledPolygon = polygon.map(([x, y]) => {
          let scaledX, scaledY
          if (flipImage) {
            // Video is not mirrored, use coordinates directly
            scaledX = x * scaleX
            scaledY = y * scaleY
          } else {
            // Video is mirrored, flip x coordinates
            scaledX = videoDisplayWidth - (x * scaleX)
            scaledY = y * scaleY
          }
          return [scaledX, scaledY]
        })

        // Draw polygon (blue for OCR)
        ctx.strokeStyle = '#0066ff'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(scaledPolygon[0][0], scaledPolygon[0][1])
        for (let i = 1; i < scaledPolygon.length; i++) {
          ctx.lineTo(scaledPolygon[i][0], scaledPolygon[i][1])
        }
        ctx.closePath()
        ctx.stroke()

        // Calculate bounding box for label positioning
        const xs = scaledPolygon.map(p => p[0])
        const ys = scaledPolygon.map(p => p[1])
        const minX = Math.min(...xs)
        const maxX = Math.max(...xs)
        const minY = Math.min(...ys)
        const maxY = Math.max(...ys)

        // Draw label background
        const labelText = `${text} ${(confidence * 100).toFixed(1)}%`
        ctx.font = 'bold 16px Arial'
        const textMetrics = ctx.measureText(labelText)
        const labelWidth = textMetrics.width + 10
        const labelHeight = 24

        // Position label at top-left of polygon
        const labelX = minX
        const labelY = Math.max(0, minY - labelHeight)

        ctx.fillStyle = 'rgba(0, 102, 255, 0.8)'
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
  }, [currentFrameDetections, currentFrameOcrDetections, flipImage])

  return (
    <div className="camera-container">
      <div className="left-section">
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
                style={{
                  transform: flipImage ? 'scaleX(1)' : 'scaleX(-1)'
                }}
              />
              <canvas 
                ref={overlayCanvasRef} 
                className="overlay-canvas"
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </>
          )}
        </div>
        
        <div className="controls-below-video">
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
          {isStreaming && (
            <button
              onClick={() => setFlipImage(!flipImage)}
              className={`control-button ${flipImage ? 'flip-active' : ''}`}
              title="Flip image horizontally"
            >
              {flipImage ? '🔄 Flip: ON' : '🔄 Flip: OFF'}
            </button>
          )}
        </div>
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

