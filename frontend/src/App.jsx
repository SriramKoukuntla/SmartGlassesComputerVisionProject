import React from 'react'
import CameraFeed from './components/CameraFeed'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Smart Glasses - Camera Feed</h1>
        <p>Visual assistance for visually impaired individuals</p>
      </header>
      <main>
        <CameraFeed />
      </main>
    </div>
  )
}

export default App

