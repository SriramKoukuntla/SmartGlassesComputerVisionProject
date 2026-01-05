import { useEffect, useRef } from 'react'
import { CANVAS_COLORS } from '../constants/config'
import { drawYOLOBox, drawOCRBox, calculateScaleFactors, scaleAndFlipX } from '../utils/canvasUtils'

/**
 * Custom hook for drawing bounding boxes on overlay canvas
 */
export const useCanvasDrawing = (videoRef, overlayCanvasRef, currentFrameDetections, currentFrameOcrDetections, flipImage) => {
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
      const { scaleX, scaleY } = calculateScaleFactors(
        videoDisplayWidth,
        videoDisplayHeight,
        videoActualWidth,
        videoActualHeight
      )

      // Clear previous drawings
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)

      // Draw YOLO bounding boxes
      currentFrameDetections.forEach((detection) => {
        const { bbox } = detection

        // Scale and flip coordinates
        let x1, x2
        if (flipImage) {
          x1 = bbox.x1 * scaleX
          x2 = bbox.x2 * scaleX
        } else {
          x1 = scaleAndFlipX(bbox.x1, scaleX, videoDisplayWidth, false)
          x2 = scaleAndFlipX(bbox.x2, scaleX, videoDisplayWidth, false)
        }
        const y1 = bbox.y1 * scaleY
        const y2 = bbox.y2 * scaleY
        const width = x2 - x1
        const height = y2 - y1

        drawYOLOBox(ctx, detection, x1, y1, width, height, CANVAS_COLORS)
      })

      // Draw OCR text boxes (polygon-based)
      currentFrameOcrDetections.forEach((ocrDetection) => {
        const [, , polygon] = ocrDetection

        if (!polygon || polygon.length !== 4) return

        // Scale polygon coordinates
        const scaledPolygon = polygon.map(([x, y]) => {
          const scaledX = scaleAndFlipX(x, scaleX, videoDisplayWidth, flipImage)
          const scaledY = y * scaleY
          return [scaledX, scaledY]
        })

        drawOCRBox(ctx, ocrDetection, scaledPolygon, CANVAS_COLORS)
      })
    }

    drawBoxes()

    // Redraw on window resize
    const handleResize = () => {
      drawBoxes()
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [currentFrameDetections, currentFrameOcrDetections, flipImage, videoRef, overlayCanvasRef])
}

