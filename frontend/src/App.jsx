import React, { useState, useEffect, useRef } from 'react';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const isProcessingRef = useRef(false);
  
  const [mode, setMode] = useState("camera"); // camera, upload
  const [scanning, setScanning] = useState(true);
  const [statusMessage, setStatusMessage] = useState("Align patient file in the box");
  const [statusType, setStatusType] = useState("active"); // active, success, error
  const [dragActive, setDragActive] = useState(false);
  
  const [patientData, setPatientData] = useState({
    name: "",
    dob: "",
    ramq: ""
  });
  
  // State variables for capture and highlights
  const [capturedImage, setCapturedImage] = useState(null);
  const [highlights, setHighlights] = useState(null);
  // Simplified latency metric
  const [totalLatency, setTotalLatency] = useState(null);

  // Audio feedback helper
  const playBeep = (freq = 880, duration = 0.15) => {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
      
      oscillator.start();
      oscillator.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio Context beep failed", e);
    }
  };

  // Start/Stop Camera based on mode & scanning state
  useEffect(() => {
    let activeStream = null;
    
    async function startCamera() {
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment", // back camera
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
        }
      } catch (err) {
        console.error("Camera access error:", err);
        setStatusMessage("Camera access denied. Please grant permission or use File Upload.");
        setStatusType("error");
      }
    }

    if (mode === "camera" && scanning) {
      startCamera();
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [mode, scanning]);

  // Continuous Scanning Loop (Camera Mode)
  useEffect(() => {
    if (!scanning || mode !== "camera") return;

    const intervalId = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Sync canvas sizing
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      if (canvas.width === 0 || canvas.height === 0) return;

      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert frame to Blob and scan
      canvas.toBlob(async (blob) => {
        if (!blob) return;

        isProcessingRef.current = true;
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
          setStatusMessage("Scanning and analyzing document...");
          const response = await fetch('http://localhost:8000/api/scan', {
            method: 'POST',
            body: formData
          });
          const data = await response.json();

          setHighlights(data.highlights);
          setTotalLatency(data.total_latency_ms);

          if (data.success) {
            playBeep(880, 0.15);
            setPatientData({
              name: data.name || "",
              dob: data.dob || "",
              ramq: data.ramq || ""
            });
            
            // Capture image snapshot from canvas
            const dataUrl = canvas.toDataURL('image/jpeg');
            setCapturedImage(dataUrl);
            
            setScanning(false);
            setStatusMessage("Patient details identified!");
            setStatusType("success");
          } else {
            // Partially update UI if we find any specific fields, but keep scanning
            if (data.name || data.dob || data.ramq) {
              setPatientData(prev => ({
                name: data.name || prev.name,
                dob: data.dob || prev.dob,
                ramq: data.ramq || prev.ramq
              }));
            }
            setStatusMessage("Searching for patient name, DOB & RAMQ...");
            setStatusType("active");
          }
        } catch (err) {
          console.error("Scan fetch error:", err);
          setStatusMessage("Unable to reach backend. Retrying...");
          setStatusType("active");
        } finally {
          isProcessingRef.current = false;
        }
      }, 'image/jpeg', 0.85);

    }, 1000); // Poll backend every 1000ms

    return () => clearInterval(intervalId);
  }, [scanning, mode]);

  // Upload/Drag-and-Drop handling
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0]);
    }
  };

  const processUploadedFile = async (file) => {
    if (!file.type.startsWith("image/")) {
      setStatusMessage("Invalid file format. Please upload an image file.");
      setStatusType("error");
      return;
    }

    setStatusMessage("Uploading and analyzing document...");
    setStatusType("active");

    // Load file as preview
    const fileReader = new FileReader();
    fileReader.onload = (event) => {
      setCapturedImage(event.target.result);
    };
    fileReader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/scan', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      setHighlights(data.highlights);
      setTotalLatency(data.total_latency_ms);

      if (data.success) {
        playBeep(880, 0.15);
        setPatientData({
          name: data.name || "",
          dob: data.dob || "",
          ramq: data.ramq || ""
        });
        setStatusMessage("Patient details identified!");
        setStatusType("success");
      } else {
        if (data.name || data.dob || data.ramq) {
          setPatientData(prev => ({
            name: data.name || prev.name,
            dob: data.dob || prev.dob,
            ramq: data.ramq || prev.ramq
          }));
          setStatusMessage("Some details identified. Check fields below.");
          setStatusType("active");
        } else {
          setStatusMessage("No patient details detected. Try another file.");
          setStatusType("error");
        }
      }
    } catch (err) {
      console.error("Upload error:", err);
      setStatusMessage("Failed to reach backend API. Try again.");
      setStatusType("error");
    }
  };

  const handleInputChange = (field, val) => {
    setPatientData(prev => ({
      ...prev,
      [field]: val
    }));
  };

  const handleReset = () => {
    setPatientData({ name: "", dob: "", ramq: "" });
    setCapturedImage(null);
    setHighlights(null);
    setTotalLatency(null);
    setScanning(true);
    setStatusMessage(mode === "camera" ? "Align patient file in the box" : "Drag and drop patient document here");
    setStatusType("active");
  };

  const handlePause = () => {
    setScanning(false);
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg');
      setCapturedImage(dataUrl);
    }
    setStatusMessage("Scanner paused.");
    setStatusType("active");
  };

  const switchMode = (newMode) => {
    setMode(newMode);
    setPatientData({ name: "", dob: "", ramq: "" });
    setCapturedImage(null);
    setHighlights(null);
    setTotalLatency(null);
    setScanning(true);
    setStatusMessage(newMode === "camera" ? "Align patient file in the box" : "Drag and drop patient document here");
    setStatusType("active");
  };

  return (
    <div className="glass-panel">
      <h1>Quebec Rx Patient Scanner</h1>
      <p className="subtitle">Instant patient details detection via document camera scanner</p>

      {/* Mode Tabs */}
      <div className="tab-container">
        <button 
          className={`tab-btn ${mode === 'camera' ? 'active' : ''}`} 
          onClick={() => switchMode('camera')}
        >
          Live Camera Scanner
        </button>
        <button 
          className={`tab-btn ${mode === 'upload' ? 'active' : ''}`} 
          onClick={() => switchMode('upload')}
        >
          Upload Image File
        </button>
      </div>

      {capturedImage ? (
        <div className="preview-container">
          <img src={capturedImage} alt="Scanned document" className="preview-image" />
          {/* Overlaid bounding boxes */}
          {highlights && Object.entries(highlights).map(([field, box]) => {
            if (!box) return null;
            return (
              <div 
                key={field}
                className={`highlight-box highlight-${field}`}
                style={{
                  left: `${box.x * 100}%`,
                  top: `${box.y * 100}%`,
                  width: `${box.w * 100}%`,
                  height: `${box.h * 100}%`
                }}
              >
                <span className="highlight-label">{field.toUpperCase()}</span>
              </div>
            );
          })}
        </div>
      ) : (
        mode === 'camera' ? (
          <div className="camera-wrapper">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="camera-preview"
            />
            
            {/* Visual Guided border frame overlays */}
            <div className="scanner-overlay">
              <div className={`scanner-frame ${statusType === 'success' ? 'success-match' : 'active'}`}>
                <div className="corner top-left"></div>
                <div className="corner top-right"></div>
                <div className="corner bottom-left"></div>
                <div className="corner bottom-right"></div>
                {scanning && <div className="laser-line"></div>}
              </div>
            </div>
          </div>
        ) : (
          <div 
            className={`upload-dropzone ${dragActive ? 'drag-active' : ''} ${statusType === 'success' ? 'success-match' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <input 
              ref={fileInputRef}
              type="file" 
              className="file-input-hidden" 
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <div className="upload-content">
              <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="upload-text">Drag and drop patient document image here</p>
              <p className="upload-hint">or click to browse local files</p>
            </div>
          </div>
        )
      )}

      <div className="status-bar">
        <span className={`status-dot ${statusType}`}></span>
        <span>{statusMessage}</span>
        {totalLatency && <span className="latency-badge">{totalLatency} ms</span>}
      </div>

      {/* Hidden Canvas for Frame Capture */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <div className="results-card">
        <div className="result-field">
          <label className="field-label">Patient Name</label>
          <input 
            type="text" 
            className="field-input"
            value={patientData.name}
            placeholder="No name detected yet"
            onChange={(e) => handleInputChange('name', e.target.value)}
          />
        </div>

        <div className="result-field">
          <label className="field-label">Date of Birth (DOB)</label>
          <input 
            type="text" 
            className="field-input"
            value={patientData.dob}
            placeholder="YYYY-MM-DD"
            onChange={(e) => handleInputChange('dob', e.target.value)}
          />
        </div>

        <div className="result-field">
          <label className="field-label">RAMQ Number</label>
          <input 
            type="text" 
            className="field-input"
            value={patientData.ramq}
            placeholder="AAAA 0000 0000"
            onChange={(e) => handleInputChange('ramq', e.target.value)}
          />
        </div>

        {mode === 'camera' ? (
          <>
            {!scanning && (
              <button className="btn btn-primary" onClick={handleReset}>
                Scan New Document
              </button>
            )}
            {scanning && (
              <button className="btn btn-secondary" onClick={handlePause}>
                Pause Scanner
              </button>
            )}
          </>
        ) : (
          <button className="btn btn-primary" onClick={handleReset}>
            Reset fields & upload new
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
