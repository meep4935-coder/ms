import uvicorn
import time
import datetime
import easyocr
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ocr_engine import extract_patient_info, get_device_gpu_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model Singleton Startup - loads once before any requests are accepted
    t0 = time.time()
    use_gpu = get_device_gpu_status()
    device_str = "CUDA GPU" if use_gpu else "CPU"
    print(f"[Lifespan] Initializing EasyOCR singleton reader on {device_str}...")
    app.state.ocr_reader = easyocr.Reader(['fr', 'en'], gpu=use_gpu, verbose=False)
    app.state.model_load_ms = round((time.time() - t0) * 1000, 2)
    app.state.startup_time = datetime.datetime.now()
    app.state.device = device_str
    app.state.total_scans = 0
    app.state.successful_scans = 0
    app.state.total_latency_ms = 0
    print(f"[Lifespan] EasyOCR singleton loaded in {app.state.model_load_ms}ms on {device_str}.")
    yield
    print("[Lifespan] Shutting down OCR API server.")

app = FastAPI(title="Quebec Patient Scanner OCR API", lifespan=lifespan)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    css_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "index.css")
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    html_head = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220%22 width=%22100%22 height=%22100%22><text y=%220.9em%22 font-size=%2290%22>🩺</text></svg>" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Quebec Patient Document OCR Scanner</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
""" + css_content + """
    </style>
  </head>
  <body>
    <div id="root"></div>

    <script type="text/babel">
      const { useState, useEffect, useRef } = React;

      function App() {
        const videoRef = useRef(null);
        const canvasRef = useRef(null);
        const fileInputRef = useRef(null);
        const isProcessingRef = useRef(false);
        
        const [mode, setMode] = useState("upload");
        const [scanning, setScanning] = useState(true);
        const [statusMessage, setStatusMessage] = useState("Drag and drop patient document here");
        const [statusType, setStatusType] = useState("active");
        const [dragActive, setDragActive] = useState(false);
        
        const [patientData, setPatientData] = useState({ name: "", dob: "", ramq: "" });
        const [capturedImage, setCapturedImage] = useState(null);
        const [highlights, setHighlights] = useState(null);
        const [totalLatency, setTotalLatency] = useState(null);
        const [diagnostics, setDiagnostics] = useState(null);
        const [showDiagnostics, setShowDiagnostics] = useState(false);
        const [healthStatus, setHealthStatus] = useState(null);

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
          } catch (e) {}
        };

        useEffect(() => {
          fetch('/api/health')
            .then(res => res.json())
            .then(data => setHealthStatus(data))
            .catch(err => console.warn("Backend health check failed:", err));
        }, []);

        useEffect(() => {
          let activeStream = null;
          async function startCamera() {
            try {
              activeStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
              });
              if (videoRef.current) {
                videoRef.current.srcObject = activeStream;
              }
            } catch (err) {
              setStatusMessage("Camera access unavailable. Please use File Upload.");
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
            if (activeStream) activeStream.getTracks().forEach(track => track.stop());
          };
        }, [mode, scanning]);

        useEffect(() => {
          if (!scanning || mode !== "camera") return;
          const intervalId = setInterval(async () => {
            if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return;
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (canvas.width !== video.videoWidth) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
            }
            if (canvas.width === 0 || canvas.height === 0) return;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(async (blob) => {
              if (!blob) return;
              isProcessingRef.current = true;
              const formData = new FormData();
              formData.append('file', blob, 'frame.jpg');
              try {
                setStatusMessage("Scanning and analyzing document...");
                const response = await fetch('/api/scan', { method: 'POST', body: formData });
                const data = await response.json();
                setHighlights(data.highlights);
                setTotalLatency(data.total_latency_ms);
                if (data.diagnostics) setDiagnostics(data.diagnostics);

                if (data.success) {
                  playBeep(880, 0.15);
                  setPatientData({ name: data.name || "", dob: data.dob || "", ramq: data.ramq || "" });
                  setCapturedImage(canvas.toDataURL('image/jpeg'));
                  setScanning(false);
                  setStatusMessage("Patient details identified!");
                  setStatusType("success");
                } else {
                  if (data.name || data.dob || data.ramq) {
                    setPatientData(prev => ({ name: data.name || prev.name, dob: data.dob || prev.dob, ramq: data.ramq || prev.ramq }));
                  }
                  setStatusMessage("Searching for patient name, DOB & RAMQ...");
                  setStatusType("active");
                }
              } catch (err) {
                setStatusMessage("Unable to reach backend. Retrying...");
                setStatusType("active");
              } finally {
                isProcessingRef.current = false;
              }
            }, 'image/jpeg', 0.85);
          }, 1000);
          return () => clearInterval(intervalId);
        }, [scanning, mode]);

        const handleDrag = (e) => {
          e.preventDefault(); e.stopPropagation();
          if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
          else if (e.type === "dragleave") setDragActive(false);
        };

        const handleDrop = (e) => {
          e.preventDefault(); e.stopPropagation(); setDragActive(false);
          if (e.dataTransfer.files && e.dataTransfer.files[0]) processUploadedFile(e.dataTransfer.files[0]);
        };

        const handleFileChange = (e) => {
          if (e.target.files && e.target.files[0]) processUploadedFile(e.target.files[0]);
        };

        const processUploadedFile = async (file) => {
          if (!file.type.startsWith("image/")) {
            setStatusMessage("Invalid file format. Please upload an image file.");
            setStatusType("error");
            return;
          }
          setStatusMessage("Uploading and analyzing document...");
          setStatusType("active");

          const fileReader = new FileReader();
          fileReader.onload = (event) => setCapturedImage(event.target.result);
          fileReader.readAsDataURL(file);

          const formData = new FormData();
          formData.append('file', file);
          try {
            const response = await fetch('/api/scan', { method: 'POST', body: formData });
            const data = await response.json();
            setHighlights(data.highlights);
            setTotalLatency(data.total_latency_ms);
            if (data.diagnostics) setDiagnostics(data.diagnostics);

            if (data.success) {
              playBeep(880, 0.15);
              setPatientData({ name: data.name || "", dob: data.dob || "", ramq: data.ramq || "" });
              setStatusMessage("Patient details identified!");
              setStatusType("success");
            } else {
              if (data.name || data.dob || data.ramq) {
                setPatientData(prev => ({ name: data.name || prev.name, dob: data.dob || prev.dob, ramq: data.ramq || prev.ramq }));
                setStatusMessage("Some details identified. Check fields below.");
                setStatusType("active");
              } else {
                setStatusMessage("No patient details detected. See diagnostics below.");
                setStatusType("error");
              }
            }
          } catch (err) {
            setStatusMessage("Failed to reach backend API. Try again.");
            setStatusType("error");
          }
        };

        const handleInputChange = (field, val) => setPatientData(prev => ({ ...prev, [field]: val }));

        const handleReset = () => {
          setPatientData({ name: "", dob: "", ramq: "" });
          setCapturedImage(null);
          setHighlights(null);
          setTotalLatency(null);
          setDiagnostics(null);
          setScanning(true);
          setStatusMessage(mode === "camera" ? "Align patient file in the box" : "Drag and drop patient document here");
          setStatusType("active");
        };

        const switchMode = (newMode) => {
          setMode(newMode);
          handleReset();
        };

        return (
          <div className="glass-panel">
            <h1>Quebec Rx Patient Scanner</h1>
            <p className="subtitle">Instant patient details detection via document camera scanner</p>

            {healthStatus && (
              <div className="server-health-bar">
                <span className="health-badge">🟢 Model Ready: {healthStatus.device}</span>
                <span className="health-stat">Load Time: {healthStatus.model_load_ms}ms</span>
                <span className="health-stat">Total Scans: {healthStatus.total_scans}</span>
                <span className="health-stat">Success Rate: {healthStatus.success_rate_pct}%</span>
              </div>
            )}

            <div className="tab-container">
              <button className={`tab-btn ${mode === 'upload' ? 'active' : ''}`} onClick={() => switchMode('upload')}>
                Upload Image File
              </button>
              <button className={`tab-btn ${mode === 'camera' ? 'active' : ''}`} onClick={() => switchMode('camera')}>
                Live Camera Scanner
              </button>
            </div>

            {capturedImage ? (
              <div className="preview-container">
                <img src={capturedImage} alt="Scanned document" className="preview-image" />
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
                  <video ref={videoRef} autoPlay playsInline muted className="camera-preview" />
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
                  onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                >
                  <input ref={fileInputRef} type="file" className="file-input-hidden" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
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

            {diagnostics?.image_quality?.quality_class === "blurry" && (
              <div className="blur-banner">
                <span className="blur-icon">⚠️</span>
                <div>
                  <strong>Low Image Quality Detected (Blur Score: {diagnostics.image_quality.blur_score})</strong>
                  <p>Image is soft or blurry. Lower confidence thresholds were applied. For best accuracy, consider retaking a clearer photo.</p>
                </div>
              </div>
            )}

            <div className="status-bar">
              <span className={`status-dot ${statusType}`}></span>
              <span>{statusMessage}</span>
              {totalLatency && <span className="latency-badge">{totalLatency} ms</span>}
            </div>

            {diagnostics?.failure_reason && (
              <div className="failure-alert">
                <span>Scan Failure Reason: <strong>{diagnostics.failure_reason}</strong></span>
              </div>
            )}

            <canvas ref={canvasRef} style={{ display: 'none' }} />

            <div className="results-card">
              <div className="result-field">
                <label className="field-label">Patient Name</label>
                <input type="text" className="field-input" value={patientData.name} placeholder="No name detected yet" onChange={(e) => handleInputChange('name', e.target.value)} />
              </div>
              <div className="result-field">
                <label className="field-label">Date of Birth (DOB)</label>
                <input type="text" className="field-input" value={patientData.dob} placeholder="YYYY-MM-DD" onChange={(e) => handleInputChange('dob', e.target.value)} />
              </div>
              <div className="result-field">
                <label className="field-label">RAMQ Number</label>
                <input type="text" className="field-input" value={patientData.ramq} placeholder="AAAA 0000 0000" onChange={(e) => handleInputChange('ramq', e.target.value)} />
              </div>

              <button className="btn btn-primary" onClick={handleReset}>
                Reset fields & scan new document
              </button>
            </div>

            {diagnostics && (
              <div className="debug-container">
                <div className="debug-header" onClick={() => setShowDiagnostics(!showDiagnostics)}>
                  <span>📊 Step-by-Step Scan Diagnostics & Pipeline Trace</span>
                  <span className="debug-toggle-btn">{showDiagnostics ? 'Hide ▲' : 'Show Details ▾'}</span>
                </div>

                {showDiagnostics && (
                  <div className="debug-body">
                    <div className="metrics-grid">
                      <div className="metric-card font-highlight">
                        <span className="metric-label">Blur Score & Quality</span>
                        <span className="metric-value">{diagnostics.image_quality?.blur_score} ({diagnostics.image_quality?.quality_class})</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">Resolution & Angle</span>
                        <span className="metric-value">{diagnostics.image_quality?.resolution?.join('x')} ({diagnostics.image_quality?.rotation_angle_deg}°)</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">Preprocessing Steps</span>
                        <span className="metric-value">{diagnostics.preprocessing?.steps_applied?.join(', ') || 'None'}</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">OCR Regions & Conf.</span>
                        <span className="metric-value">{diagnostics.ocr_summary?.regions_above_threshold} / {diagnostics.ocr_summary?.regions_total} ({diagnostics.ocr_summary?.avg_confidence})</span>
                      </div>
                    </div>

                    <div className="latency-section">
                      <div className="panel-title">Latency Breakdown (ms)</div>
                      <div className="latency-grid">
                        <div className="lat-item">Load: <span>{diagnostics.latency_breakdown?.image_load}ms</span></div>
                        <div className="lat-item">Preprocess: <span>{diagnostics.latency_breakdown?.preprocessing}ms</span></div>
                        <div className="lat-item">OCR: <span>{diagnostics.latency_breakdown?.ocr_inference}ms</span></div>
                        <div className="lat-item">Regex: <span>{diagnostics.latency_breakdown?.regex_parsing}ms</span></div>
                        <div className="lat-item total">Total: <span>{diagnostics.latency_breakdown?.total}ms</span></div>
                      </div>
                    </div>

                    <div className="log-panel">
                      <div className="panel-title">Pipeline Trace Log (Step-by-Step Math View)</div>
                      <div className="log-entries">
                        {diagnostics.pipeline_trace?.map((step, idx) => (
                          <div className="log-entry" key={idx}>
                            <span className="log-index">[{step.t_ms}ms]</span>
                            <span className="log-event">{step.event}:</span>
                            <span className="log-text">{step.detail}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      }

      ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
  </body>
</html>"""
    return HTMLResponse(content=html_head)

@app.get("/api/health")
async def health_check():
    uptime = int((datetime.datetime.now() - app.state.startup_time).total_seconds()) if hasattr(app.state, "startup_time") else 0
    total_scans = getattr(app.state, "total_scans", 0)
    succ_scans = getattr(app.state, "successful_scans", 0)
    tot_lat = getattr(app.state, "total_latency_ms", 0.0)
    
    success_rate = round((succ_scans / total_scans * 100), 1) if total_scans > 0 else 100.0
    avg_latency = round(tot_lat / total_scans, 2) if total_scans > 0 else 0.0

    return {
        "status": "ready" if hasattr(app.state, "ocr_reader") else "initializing",
        "model": "EasyOCR [fr, en]",
        "device": getattr(app.state, "device", "unknown"),
        "model_load_ms": getattr(app.state, "model_load_ms", 0),
        "uptime_seconds": uptime,
        "total_scans": total_scans,
        "successful_scans": succ_scans,
        "success_rate_pct": success_rate,
        "avg_latency_ms": avg_latency
    }

@app.post("/api/scan")
async def scan_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        res = extract_patient_info(contents, reader=app.state.ocr_reader)
        
        # Track aggregate server metrics
        app.state.total_scans += 1
        if res.get("success"):
            app.state.successful_scans += 1
        app.state.total_latency_ms += res.get("total_latency_ms", 0)
        
        return res
    except Exception as e:
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
