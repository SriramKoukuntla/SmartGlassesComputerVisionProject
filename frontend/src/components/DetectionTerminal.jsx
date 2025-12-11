import React, { useEffect, useRef } from 'react'
import './DetectionTerminal.css'

const DetectionTerminal = ({ detections, isProcessing }) => {
  const terminalRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom when new detections arrive
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [detections])

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
        {detections.length === 0 ? (
          <div className="terminal-line">
            <span className="terminal-prompt">$</span>
            <span className="terminal-text">Waiting for detections...</span>
          </div>
        ) : (
          detections.map((detection, index) => (
            <div key={index} className="terminal-line">
              <span className="terminal-timestamp">[{formatTimestamp()}]</span>
              <span className="terminal-class">{detection.class}</span>
              <span className="terminal-confidence">{detection.confidence}%</span>
            </div>
          ))
        )}
      </div>
      <div className="terminal-footer">
        <span className="terminal-count">
          Total Detections: {detections.length}
        </span>
      </div>
    </div>
  )
}

export default DetectionTerminal

