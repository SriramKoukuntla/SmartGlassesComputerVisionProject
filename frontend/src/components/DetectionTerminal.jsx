import React, { useEffect, useRef } from 'react'
import './DetectionTerminal.css'

const DetectionTerminal = ({ detections, ocrTextDetections = [], isProcessing, backendLogs = [] }) => {
  const terminalRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom when new detections arrive
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [detections, ocrTextDetections, backendLogs])

  const formatTimestamp = () => {
    const now = new Date()
    return now.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    })
  }

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
        {detections.length === 0 && ocrTextDetections.length === 0 && backendLogs.length === 0 ? (
          <div className="terminal-line">
            <span className="terminal-prompt">$</span>
            <span className="terminal-text">Waiting for detections...</span>
          </div>
        ) : (
          <>
            {/* Backend logs */}
            {backendLogs.map((log, index) => {
              const isError = log.includes('[ERROR]')
              const isWarning = log.includes('[WARNING]')
              return (
                <div key={`log-${index}`} className={`terminal-line ${isError ? 'log-error' : isWarning ? 'log-warning' : 'log-info'}`}>
                  <span className="terminal-prompt">LOG:</span>
                  <span className="terminal-text">{log}</span>
                </div>
              )
            })}
            {/* YOLO detections */}
            {detections.map((detection, index) => (
              <div key={`yolo-${index}`} className="terminal-line">
                <span className="terminal-timestamp">[{formatTimestamp()}]</span>
                <span className="terminal-class">{detection.class}</span>
                <span className="terminal-confidence">{detection.confidence}%</span>
              </div>
            ))}
            {/* OCR detections */}
            {ocrTextDetections.map((ocrDetection, index) => {
              // OCR detection format: [text, confidence, polygon]
              const text = Array.isArray(ocrDetection) ? ocrDetection[0] : ocrDetection.text || 'Unknown'
              const confidence = Array.isArray(ocrDetection) ? (ocrDetection[1] * 100).toFixed(1) : (ocrDetection.confidence || 0).toFixed(1)
              return (
                <div key={`ocr-${index}`} className="terminal-line">
                  <span className="terminal-timestamp">[{formatTimestamp()}]</span>
                  <span className="terminal-prompt">OCR:</span>
                  <span className="terminal-text">{text}</span>
                  <span className="terminal-confidence">{confidence}%</span>
                </div>
              )
            })}
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

