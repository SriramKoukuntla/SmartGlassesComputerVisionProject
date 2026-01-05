/**
 * Utility functions for formatting data
 */

/**
 * Formats a timestamp for display
 * @returns {string} Formatted timestamp string
 */
export const formatTimestamp = () => {
  const now = new Date()
  return now.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
  })
}

/**
 * Extracts text from OCR detection (handles both array and object formats)
 * @param {Array|Object} ocrDetection - OCR detection
 * @returns {string} Extracted text
 */
export const extractOCRText = (ocrDetection) => {
  if (Array.isArray(ocrDetection)) {
    return ocrDetection[0] || 'Unknown'
  }
  return ocrDetection.text || 'Unknown'
}

/**
 * Extracts confidence from OCR detection (handles both array and object formats)
 * @param {Array|Object} ocrDetection - OCR detection
 * @returns {number} Confidence value (0-100)
 */
export const extractOCRConfidence = (ocrDetection) => {
  if (Array.isArray(ocrDetection)) {
    return (ocrDetection[1] * 100).toFixed(1)
  }
  return (ocrDetection.confidence || 0).toFixed(1)
}

