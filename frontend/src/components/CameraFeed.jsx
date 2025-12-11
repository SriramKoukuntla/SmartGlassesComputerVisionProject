import React, { useRef, useEffect, useState } from 'react'
import './CameraFeed.css'

const CameraFeed = () => {
  const videoRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

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
    }
  }, [selectedDeviceId])

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
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
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="camera-video"
          />
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
      </div>
    </div>
  )
}

export default CameraFeed

