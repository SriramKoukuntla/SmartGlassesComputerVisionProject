import React, { useEffect, useRef } from 'react'
import { formatTimestamp, extractOCRText, extractOCRConfidence } from '../utils/formatUtils'
import './DetectionTerminal.css'

const DetectionTerminal = ({
  detections = [],
  ocrTextDetections = [],
  isProcessing = false,
  backendLogs = [],
}) => {
  const terminalRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom when new detections arrive
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [detections, ocrTextDetections, backendLogs])

  const renderBackendLogs = () => {
    if (backendLogs.length === 0) return null

    return backendLogs.map((log, index) => {
      const isError = log.includes('[ERROR]')
      const isWarning = log.includes('[WARNING]')
      return (
        <div
          key={`log-${index}`}
          className={`terminal-line ${isError ? 'log-error' : isWarning ? 'log-warning' : 'log-info'}`}
        >
          <span className="terminal-prompt">LOG:</span>
          <span className="terminal-text">{log}</span>
        </div>
      )
    })
  }

  const renderYOLODetections = () => {
    if (detections.length === 0) return null

    return detections.map((detection, index) => (
      <div key={`yolo-${index}`} className="terminal-line">
        <span className="terminal-timestamp">[{formatTimestamp()}]</span>
        <span className="terminal-class">{detection.class}</span>
        <span className="terminal-confidence">{detection.confidence}%</span>
      </div>
    ))
  }

  const renderOCRDetections = () => {
    if (ocrTextDetections.length === 0) return null

    return ocrTextDetections.map((ocrDetection, index) => {
      const text = extractOCRText(ocrDetection)
      const confidence = extractOCRConfidence(ocrDetection)

      return (
        <div key={`ocr-${index}`} className="terminal-line">
          <span className="terminal-timestamp">[{formatTimestamp()}]</span>
          <span className="terminal-prompt">OCR:</span>
          <span className="terminal-text">{text}</span>
          <span className="terminal-confidence">{confidence}%</span>
        </div>
      )
    })
  }

  const hasContent = detections.length > 0 || ocrTextDetections.length > 0 || backendLogs.length > 0

  return (
    <div className="detection-terminal">
      <div className="terminal-header">
        <div className="terminal-title">
          <span>YOLOv11n Object Detection</span>
        </div>
        <div className="terminal-status">
          {isProcessing ? (
            <span className="status-processing">Processing...</span>
          ) : (
            <span className="status-ready">Ready</span>
          )}
        </div>
      </div>
      <div className="terminal-body" ref={terminalRef}>
        {!hasContent ? (
          <div className="terminal-line">
            <span className="terminal-prompt">$</span>
            <span className="terminal-text">Waiting for detections...</span>
          </div>
        ) : (
          <>
            {renderBackendLogs()}
            {renderYOLODetections()}
            {renderOCRDetections()}
          </>
        )}
      </div>
      <div className="terminal-footer">
        <span className="terminal-count">
          Objects: {detections.length} | Text: {ocrTextDetections.length} | Logs: {backendLogs.length}
        </span>
      </div>
    </div>
  )
}

export default DetectionTerminal
