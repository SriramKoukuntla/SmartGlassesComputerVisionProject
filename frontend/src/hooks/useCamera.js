import { useState, useEffect, useRef, useCallback } from 'react'
import { CAMERA_CONFIG } from '../constants/config'

/**
 * Custom hook for managing camera stream and device selection
 */
export const useCamera = () => {
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const cameraStoppedRef = useRef(false)

  // Get available video input devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        // Request permission to access devices
        await navigator.mediaDevices.getUserMedia({ video: true })

        // Enumerate devices
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

  // Stop camera stream
  const stopCamera = useCallback(() => {
    cameraStoppedRef.current = true

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop()
        track.enabled = false
      })
      streamRef.current = null
    }

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject
      stream.getTracks().forEach(track => {
        track.stop()
        track.enabled = false
      })
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }, [])

  // Start camera stream
  const startCamera = useCallback(async () => {
    if (!selectedDeviceId) {
      setError('Please select a camera device.')
      return
    }

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

      if (videoRef.current?.srcObject) {
        const existingStream = videoRef.current.srcObject
        existingStream.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
        videoRef.current.srcObject = null
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedDeviceId },
          width: { ideal: CAMERA_CONFIG.IDEAL_WIDTH },
          height: { ideal: CAMERA_CONFIG.IDEAL_HEIGHT },
        },
      })

      streamRef.current = stream

      // Add event listeners to track stream state
      stream.getTracks().forEach(track => {
        track.onended = () => {
          console.warn('Camera track ended:', track.kind)
          if (!cameraStoppedRef.current) {
            setError('Camera track ended. Trying to reconnect...')
            setIsStreaming(false)
            setTimeout(() => {
              if (!cameraStoppedRef.current) {
                setSelectedDeviceId(prev => prev)
              }
            }, 1000)
          }
        }

        track.onerror = (e) => {
          console.error('Camera track error:', e)
          if (!cameraStoppedRef.current) {
            setError('Camera track error. Trying to reconnect...')
            setIsStreaming(false)
          }
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setError(null)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError(`Unable to access camera: ${err.message}. Please ensure you have granted camera permissions.`)
      setIsStreaming(false)
    }
  }, [selectedDeviceId])

  // Handle video stream events
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setIsStreaming(true)
      setError(null)
    }

    const handleError = () => {
      setError('Video stream error. Trying to reconnect...')
      setIsStreaming(false)
      setTimeout(() => {
        if (!cameraStoppedRef.current && selectedDeviceId) {
          setSelectedDeviceId(prev => prev)
        }
      }, 1000)
    }

    const handleEnded = () => {
      setError('Camera stream ended. Trying to reconnect...')
      setIsStreaming(false)
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

  // Auto-start camera when device is selected
  useEffect(() => {
    let stream = null
    let isMounted = true

    const startCameraAuto = async () => {
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

        if (videoRef.current?.srcObject) {
          const existingStream = videoRef.current.srcObject
          existingStream.getTracks().forEach(track => {
            track.stop()
            track.enabled = false
          })
          videoRef.current.srcObject = null
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedDeviceId },
            width: { ideal: CAMERA_CONFIG.IDEAL_WIDTH },
            height: { ideal: CAMERA_CONFIG.IDEAL_HEIGHT },
          },
        })

        if (!isMounted) {
          stream.getTracks().forEach(track => track.stop())
          return
        }

        streamRef.current = stream

        stream.getTracks().forEach(track => {
          track.onended = () => {
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

          track.onerror = () => {
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
          cameraStoppedRef.current = false
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
      startCameraAuto()
    }

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
    }
  }, [selectedDeviceId])

  const handleDeviceChange = useCallback((event) => {
    setSelectedDeviceId(event.target.value)
  }, [])

  return {
    devices,
    selectedDeviceId,
    isStreaming,
    error,
    videoRef,
    setSelectedDeviceId,
    startCamera,
    stopCamera,
    handleDeviceChange,
  }
}

