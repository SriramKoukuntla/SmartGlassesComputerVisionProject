/**
 * Utility functions for canvas operations
 */

/**
 * Draws a YOLO detection bounding box on the canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} detection - Detection object with bbox, class, and confidence
 * @param {number} x1 - Scaled x1 coordinate
 * @param {number} y1 - Scaled y1 coordinate
 * @param {number} width - Scaled width
 * @param {number} height - Scaled height
 * @param {Object} colors - Color configuration
 */
export const drawYOLOBox = (ctx, detection, x1, y1, width, height, colors) => {
  const { class: className, confidence } = detection

  // Draw bounding box
  ctx.strokeStyle = colors.YOLO_BOX
  ctx.lineWidth = 3
  ctx.strokeRect(x1, y1, width, height)

  // Draw label
  const labelText = `${className} ${confidence.toFixed(1)}%`
  ctx.font = 'bold 16px Arial'
  const textMetrics = ctx.measureText(labelText)
  const labelWidth = textMetrics.width + 10
  const labelHeight = 24

  const labelX = x1
  const labelY = Math.max(0, y1 - labelHeight)

  ctx.fillStyle = colors.YOLO_LABEL_BG
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight)

  ctx.fillStyle = colors.YOLO_LABEL_TEXT
  ctx.fillText(labelText, labelX + 5, labelY + 18)
}

/**
 * Draws an OCR detection polygon on the canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array} ocrDetection - OCR detection [text, confidence, polygon]
 * @param {Array} scaledPolygon - Scaled polygon coordinates
 * @param {Object} colors - Color configuration
 */
export const drawOCRBox = (ctx, ocrDetection, scaledPolygon, colors) => {
  const [text, confidence] = ocrDetection

  // Draw polygon
  ctx.strokeStyle = colors.OCR_BOX
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(scaledPolygon[0][0], scaledPolygon[0][1])
  for (let i = 1; i < scaledPolygon.length; i++) {
    ctx.lineTo(scaledPolygon[i][0], scaledPolygon[i][1])
  }
  ctx.closePath()
  ctx.stroke()

  // Calculate bounding box for label
  const xs = scaledPolygon.map(p => p[0])
  const ys = scaledPolygon.map(p => p[1])
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)

  // Draw label
  const labelText = `${text} ${(confidence * 100).toFixed(1)}%`
  ctx.font = 'bold 16px Arial'
  const textMetrics = ctx.measureText(labelText)
  const labelWidth = textMetrics.width + 10
  const labelHeight = 24

  const labelX = minX
  const labelY = Math.max(0, minY - labelHeight)

  ctx.fillStyle = colors.OCR_LABEL_BG
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight)

  ctx.fillStyle = colors.OCR_LABEL_TEXT
  ctx.fillText(labelText, labelX + 5, labelY + 18)
}

/**
 * Calculates scale factors for converting video coordinates to display coordinates
 * @param {number} videoDisplayWidth - Display width of video element
 * @param {number} videoDisplayHeight - Display height of video element
 * @param {number} videoActualWidth - Actual video width
 * @param {number} videoActualHeight - Actual video height
 * @returns {Object} Scale factors { scaleX, scaleY }
 */
export const calculateScaleFactors = (videoDisplayWidth, videoDisplayHeight, videoActualWidth, videoActualHeight) => {
  return {
    scaleX: videoDisplayWidth / videoActualWidth,
    scaleY: videoDisplayHeight / videoActualHeight,
  }
}

/**
 * Scales and optionally flips x coordinate based on flipImage setting
 * @param {number} x - Original x coordinate
 * @param {number} scaleX - X scale factor
 * @param {number} videoDisplayWidth - Display width
 * @param {boolean} flipImage - Whether image is flipped
 * @returns {number} Scaled and optionally flipped x coordinate
 */
export const scaleAndFlipX = (x, scaleX, videoDisplayWidth, flipImage) => {
  if (flipImage) {
    return x * scaleX
  }
  return videoDisplayWidth - (x * scaleX)
}

