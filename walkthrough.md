# Quebec Rx Patient Scanner - Walkthrough

The patient document OCR scanner tool has been successfully created under the directory:
[med-scanner/](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner)

## Changes Made

### 1. Backend Service
- [ocr_engine.py](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/backend/ocr_engine.py): Contains RAMQ formatting check logic, Quebec-specific date rules validation, and EasyOCR extractor.
- [main.py](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/backend/main.py): Exposes FastAPI endpoint `/api/scan` to receive camera frame blobs.
- [requirements.txt](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/backend/requirements.txt): Lists python requirements.
- [test_ocr.py](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/backend/test_ocr.py): Local tests for verifying Quebec RAMQ-to-DOB parsing algorithm.

### 2. Frontend React Application (Vite-powered)
- [index.html](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/frontend/index.html): Document setup.
- [App.jsx](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/frontend/src/App.jsx): Grabs video camera frame blobs and sends them to the backend endpoint every 1000ms until Name, DOB, and RAMQ are matched. Includes Web Audio API beep sound feedback on successful recognition.
- [index.css](file:///c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/frontend/src/index.css): Beautifully designed dark-themed clinician dashboard with glowing scanner frames and sweep lines.

---

## Instructions to Run

To run the application, make sure Node.js and Python are installed on your machine.

### Step A: Start the Backend (FastAPI)
1. Open terminal and navigate to the backend folder:
   ```bash
   cd c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   python main.py
   ```
   *Note: On the first run, EasyOCR will automatically download standard model weights (~100-150MB) for French and English character detection.*

### Step B: Start the Frontend (Vite + React)
1. Open another terminal and navigate to the frontend folder:
   ```bash
   cd c:/Users/WongK2/.gemini/antigravity/scratch/cleantech-scraper-extension/med-scanner/frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Run Vite server in development mode:
   ```bash
   npm run dev
   ```
4. Access the scanner via your browser or scan the network address with your mobile device to test it live with your camera.
