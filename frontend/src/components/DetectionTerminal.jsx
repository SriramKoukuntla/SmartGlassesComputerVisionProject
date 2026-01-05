import React, { useEffect, useRef, useMemo } from 'react'
import { formatTimestamp, extractOCRText, extractOCRConfidence } from '../utils/formatUtils'
import './DetectionTerminal.css'

const DetectionTerminal = ({
  detections = [],
  ocrTextDetections = [],
  isProcessing = false,
  backendLogs = [],
  isConnected = false,
}) => {
  const terminalRef = useRef(null)

  // Parse timestamp from log string (format: [123.456] message)
  const parseLogTimestamp = (logText) => {
    const match = logText.match(/^\[(\d+\.?\d*)\]/)
    if (match) {
      return parseFloat(match[1]) * 1000 // Convert seconds to milliseconds
    }
    return null
  }

  // Create unified event list with all events sorted by timestamp
  const unifiedEvents = useMemo(() => {
    const events = []

    // Add logs
    backendLogs.forEach((log, index) => {
      const logText = typeof log === 'string' ? log : log.text
      const timestamp = typeof log === 'object' && log.timestamp 
        ? log.timestamp 
        : parseLogTimestamp(logText) || Date.now()
      
      events.push({
        type: 'log',
        timestamp,
        data: logText,
        key: `log-${index}-${timestamp}`
      })
    })

    // Add YOLO detections
    detections.forEach((detection, index) => {
      const timestamp = detection.timestamp || Date.now()
      events.push({
        type: 'yolo',
        timestamp,
        data: detection,
        key: `yolo-${index}-${timestamp}`
      })
    })

    // Add OCR detections
    ocrTextDetections.forEach((ocrDetection, index) => {
      const timestamp = ocrDetection.timestamp || Date.now()
      events.push({
        type: 'ocr',
        timestamp,
        data: ocrDetection,
        key: `ocr-${index}-${timestamp}`
      })
    })

    // Sort by timestamp (oldest first)
    return events.sort((a, b) => a.timestamp - b.timestamp)
  }, [detections, ocrTextDetections, backendLogs])

  useEffect(() => {
    // Auto-scroll to bottom when new detections arrive
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [unifiedEvents])

  const renderEvent = (event) => {
    const timestamp = new Date(event.timestamp)
    const formattedTime = timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    })

    switch (event.type) {
      case 'log': {
        const logText = event.data
        const isError = logText.includes('[ERROR]')
        const isWarning = logText.includes('[WARNING]')
        return (
          <div
            key={event.key}
            className={`terminal-line ${isError ? 'log-error' : isWarning ? 'log-warning' : 'log-info'}`}
          >
            <span className="terminal-timestamp">[{formattedTime}]</span>
            <span className="terminal-prompt">LOG:</span>
            <span className="terminal-text">{logText}</span>
          </div>
        )
      }
      case 'yolo': {
        const detection = event.data
        return (
          <div key={event.key} className="terminal-line">
            <span className="terminal-timestamp">[{formattedTime}]</span>
            <span className="terminal-class">{detection.class}</span>
            <span className="terminal-confidence">{detection.confidence}%</span>
          </div>
        )
      }
      case 'ocr': {
        const ocrDetection = event.data
        const text = extractOCRText(ocrDetection)
        const confidence = extractOCRConfidence(ocrDetection)
        return (
          <div key={event.key} className="terminal-line">
            <span className="terminal-timestamp">[{formattedTime}]</span>
            <span className="terminal-prompt">OCR:</span>
            <span className="terminal-text">{text}</span>
            <span className="terminal-confidence">{confidence}%</span>
          </div>
        )
      }
      default:
        return null
    }
  }

  const hasContent = unifiedEvents.length > 0

  return (
    <div className="detection-terminal">
      <div className="terminal-header">
        <div className="terminal-title">
          <span>YOLOv11n Object Detection</span>
        </div>
        <div className="terminal-status">
          {!isConnected ? (
            <span className="status-disconnected">Disconnected</span>
          ) : isProcessing ? (
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
          unifiedEvents.map(event => renderEvent(event))
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
