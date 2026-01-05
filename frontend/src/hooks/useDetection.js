import { useState, useCallback, useRef, useEffect } from 'react'
import { API_CONFIG, DETECTION_LIMITS } from '../constants/config'

/**
 * Custom hook for managing object detection via WebSocket
 */
export const useDetection = (videoRef, canvasRef, flipImage, requestInterval) => {
  const [detections, setDetections] = useState([])
  const [currentFrameDetections, setCurrentFrameDetections] = useState([])
  const [ocrTextDetections, setOcrTextDetections] = useState([])
  const [currentFrameOcrDetections, setCurrentFrameOcrDetections] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [backendLogs, setBackendLogs] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  const detectionIntervalRef = useRef(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const isProcessingRef = useRef(false)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5
  const reconnectDelay = 3000 // 3 seconds

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // Already connected
    }

    try {
      const wsUrl = `${API_CONFIG.WS_URL}${API_CONFIG.ENDPOINTS.WS_VIDEO}`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        setBackendLogs(prev => {
          const newLogs = [...prev, {
            text: '[INFO] WebSocket connected',
            timestamp: Date.now()
          }]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)

          if (message.type === 'connected') {
            console.log('WebSocket connection confirmed:', message.message)
          } else if (message.type === 'result') {
            const data = message.data

            // Extract logs from backend and add timestamp if not present
            if (data.logs && Array.isArray(data.logs)) {
              const arrivalTime = Date.now()
              const logsWithTimestamp = data.logs.map(log => ({
                text: log,
                timestamp: arrivalTime
              }))
              setBackendLogs(prev => {
                const newLogs = [...prev, ...logsWithTimestamp]
                return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
              })
            }

            // Extract YOLO detections and add timestamp
            const yoloDetections = data.yolo_result?.detections || []
            const ocrResults = data.paddleocr_result || null
            const ocrTextDetections = ocrResults?.text_detections || []
            const arrivalTime = Date.now()

            // Update detections state with timestamps
            if (yoloDetections.length > 0) {
              const detectionsWithTimestamp = yoloDetections.map(det => ({
                ...det,
                timestamp: arrivalTime
              }))
              setDetections(prev => {
                const newDetections = [...prev, ...detectionsWithTimestamp]
                return newDetections.slice(-DETECTION_LIMITS.MAX_DETECTIONS)
              })
              setCurrentFrameDetections(yoloDetections)
            } else {
              setCurrentFrameDetections([])
            }

            // Update OCR text detections state with timestamps
            if (ocrTextDetections.length > 0) {
              const ocrWithTimestamp = ocrTextDetections.map(ocr => ({
                ...ocr,
                timestamp: arrivalTime
              }))
              setOcrTextDetections(prev => {
                const newOcrDetections = [...prev, ...ocrWithTimestamp]
                return newOcrDetections.slice(-DETECTION_LIMITS.MAX_OCR_DETECTIONS)
              })
              setCurrentFrameOcrDetections(ocrTextDetections)
            } else {
              setCurrentFrameOcrDetections([])
            }

            setIsProcessing(false)
            isProcessingRef.current = false
          } else if (message.type === 'error') {
            console.error('WebSocket error:', message.message)
            const errorLog = {
              text: `[ERROR] ${message.message}`,
              timestamp: Date.now()
            }
            setBackendLogs(prev => {
              const newLogs = [...prev, errorLog]
              return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
            })
            setIsProcessing(false)
            isProcessingRef.current = false
          } else if (message.type === 'pong') {
            // Keepalive response, no action needed
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setBackendLogs(prev => {
          const newLogs = [...prev, {
            text: '[ERROR] WebSocket connection error',
            timestamp: Date.now()
          }]
          return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
        })
        setIsConnected(false)
      }

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason)
        setIsConnected(false)
        wsRef.current = null

        // Attempt reconnection if not intentionally closed
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1
          const delay = reconnectDelay * reconnectAttemptsRef.current
          setBackendLogs(prev => {
            const newLogs = [...prev, {
              text: `[INFO] Attempting to reconnect in ${delay / 1000}s (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`,
              timestamp: Date.now()
            }]
            return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
          })
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket()
          }, delay)
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setBackendLogs(prev => {
            const newLogs = [...prev, {
              text: '[ERROR] Max reconnection attempts reached. Please refresh the page.',
              timestamp: Date.now()
            }]
            return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
          })
        }
      }
    } catch (error) {
      console.error('Error creating WebSocket:', error)
      setBackendLogs(prev => {
        const newLogs = [...prev, {
          text: `[ERROR] Failed to create WebSocket connection: ${error.message}`,
          timestamp: Date.now()
        }]
        return newLogs.slice(-DETECTION_LIMITS.MAX_LOGS)
      })
    }
  }, [])

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    setIsConnected(false)
  }, [])

  // Capture frame and send to backend via WebSocket
  const captureAndDetect = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return

    const video = videoRef.current

    // Check if video is ready
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return
    }

    // Check WebSocket connection
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      if (!isConnected) {
        connectWebSocket()
      }
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

      // Send frame via WebSocket
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        image: imageData
      }))
    } catch (err) {
      console.error('Error capturing frame:', err)
      setIsProcessing(false)
      isProcessingRef.current = false
    }
  }, [flipImage, videoRef, canvasRef, isConnected, connectWebSocket])

  // Start detection interval
  const startDetectionInterval = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
    }

    // Ensure WebSocket is connected
    if (!isConnected) {
      connectWebSocket()
    }

    detectionIntervalRef.current = setInterval(() => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        captureAndDetect()
      }
    }, requestInterval)
  }, [requestInterval, captureAndDetect, videoRef, isConnected, connectWebSocket])

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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetectionInterval()
      disconnectWebSocket()
    }
  }, [stopDetectionInterval, disconnectWebSocket])

  // Connect WebSocket when component mounts or when explicitly requested
  useEffect(() => {
    connectWebSocket()
    return () => {
      disconnectWebSocket()
    }
  }, [connectWebSocket, disconnectWebSocket])

  return {
    detections,
    currentFrameDetections,
    ocrTextDetections,
    currentFrameOcrDetections,
    isProcessing,
    backendLogs,
    isConnected,
    captureAndDetect,
    startDetectionInterval,
    stopDetectionInterval,
    clearDetections,
    connectWebSocket,
    disconnectWebSocket,
    detectionIntervalRef,
  }
}

