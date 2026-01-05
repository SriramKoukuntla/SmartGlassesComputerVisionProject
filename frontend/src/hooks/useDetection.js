import { useState, useCallback, useRef } from 'react'
import { API_CONFIG, DETECTION_LIMITS } from '../constants/config'

/**
 * Custom hook for managing object detection API calls
 */
export const useDetection = (videoRef, canvasRef, flipImage, requestInterval) => {
  const [detections, setDetections] = useState([])
  const [currentFrameDetections, setCurrentFrameDetections] = useState([])
  const [ocrTextDetections, setOcrTextDetections] = useState([])
  const [currentFrameOcrDetections, setCurrentFrameOcrDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [backendLogs, setBackendLogs] = useState([])
  const detectionIntervalRef = useRef(null)
  const isProcessingRef = useRef(false)

  // Capture frame and send to backend for detection
  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return

    const video = videoRef.current

    // Check if video is ready
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
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
      isProcessingRef.current = true

      // Send to backend API
      let response
      try {
        response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.INPUT_IMAGE}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        })
      } catch (fetchError) {
        console.error('Network error:', fetchError)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] Network error: ${fetchError.message}. Is the backend API server running on ${API_CONFIG.BASE_URL}?`]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
        throw fetchError
      }

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API error:', response.status, errorText)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] API request failed: ${response.status} ${response.statusText}`]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
        throw new Error(`Detection request failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()

      // Log any errors from backend
      if (data.error) {
        console.error('Backend error:', data.error)
        setBackendLogs(prev => {
          const newLogs = [...prev, `[ERROR] ${data.error}`]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
      }

      // Extract logs from backend
      if (data.logs && Array.isArray(data.logs)) {
        setBackendLogs(prev => {
          const newLogs = [...prev, ...data.logs]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
      }

      // Extract YOLO detections
      const yoloDetections = data.yolo_result?.detections || []
      const ocrResults = data.paddleocr_result || null
      const ocrTextDetections = ocrResults?.text_detections || []

      // Update detections state
      if (yoloDetections.length > 0) {
        setDetections(prev => {
          const newDetections = [...prev, ...yoloDetections]
          return newDetections.slice(-DETECTION_LIMITS.MAX_DETECTIONS)
        })
        setCurrentFrameDetections(yoloDetections)
      } else {
        setCurrentFrameDetections([])
      }

      // Update OCR text detections state
      if (ocrTextDetections.length > 0) {
        setOcrTextDetections(prev => {
          const newOcrDetections = [...prev, ...ocrTextDetections]
          return newOcrDetections.slice(-DETECTION_LIMITS.MAX_OCR_DETECTIONS)
        })
        setCurrentFrameOcrDetections(ocrTextDetections)
      } else {
        setCurrentFrameOcrDetections([])
      }
    } catch (err) {
      console.error('Error detecting objects:', err)
    } finally {
      setIsProcessing(false)
      isProcessingRef.current = false
    }
  }, [flipImage, videoRef, canvasRef]) // Removed isProcessing to prevent unnecessary recreations

  // Start detection interval
  const startDetectionInterval = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
    }

    detectionIntervalRef.current = setInterval(() => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        captureAndDetect()
      }
    }, requestInterval)
  }, [requestInterval, captureAndDetect, videoRef])

  // Stop detection interval
  const stopDetectionInterval = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
  }, [])

  // Clear all detections and logs
  const clearDetections = useCallback(() => {
    setCurrentFrameDetections([])
    setCurrentFrameOcrDetections([])
    setDetections([])
    setOcrTextDetections([])
    setBackendLogs([])
  }, [])

  return {
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
    detectionIntervalRef,
  }
}

