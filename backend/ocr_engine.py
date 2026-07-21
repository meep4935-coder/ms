import re
import datetime
import time
import json
import os
import easyocr
import numpy as np
from PIL import Image
import cv2
import concurrent.futures
import json
from pathlib import Path

# Initialize the EasyOCR reader. 
# We include French ('fr') and English ('en') since both are prevalent in Quebec clinical settings.
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['fr', 'en'], gpu=False)
print("EasyOCR reader initialized.")

# Path to the logs folder
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def log_scan_details(log_entry: dict):
    try:
        timestamp_ms = int(time.time() * 1000)
        filename = f"scan_{timestamp_ms}.json"
        filepath = os.path.join(LOGS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=4)
        print(f"Logged ocr scan details to {filepath}")
    except Exception as e:
        print(f"Error writing scan log: {e}")

def parse_ramq_dob_info(ramq_str: str):
    """
    Extract DOB and gender info from RAMQ string.
    Quebec RAMQ format: AAAA YYMMDD XX
    YY: Birth year
    MM: Birth month. Females have 50 added to their birth month (51-62).
    DD: Birth day
    """
    clean_ramq = re.sub(r'[^A-Z0-9]', '', ramq_str.upper())
    if len(clean_ramq) != 12:
        return None
    
    initials = clean_ramq[:4]
    yy_str = clean_ramq[4:6]
    mm_val = int(clean_ramq[6:8])
    dd_str = clean_ramq[8:10]
    
    # Check gender
    is_female = False
    if mm_val > 50:
        is_female = True
        mm_val -= 50
        
    if not (1 <= mm_val <= 12):
        return None
        
    # Determine full year (assuming RAMQ is for living patients, birth year is likely 19xx or 20xx)
    current_year = datetime.datetime.now().year
    current_yy = current_year % 100
    
    yy = int(yy_str)
    # If YY is greater than current YY, it's likely 19xx, otherwise 20xx
    if yy > current_yy:
        year = 1900 + yy
    else:
        year = 2000 + yy
        
    try:
        dob = datetime.date(year, mm_val, int(dd_str))
        return {
            "dob": dob.strftime("%Y-%m-%d"),
            "gender": "F" if is_female else "M",
            "initials": initials
        }
    except ValueError:
        return None

def extract_patient_info(image_bytes: bytes):
    """
    Process image bytes, run OCR, and extract patient fields with bounding boxes and timing metrics.
    """
    t_start = time.time()
    
    # Load image
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    img_width, img_height = image.size
    image_np = np.array(image)
    
    # Load OCR parallel configuration
    config_path = Path(__file__).with_name('ocr_parallel_config.json')
    if config_path.is_file():
        with open(config_path, 'r') as cfg_file:
            cfg = json.load(cfg_file)
    else:
        cfg = {
            "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "confidence_threshold": 0.3
        }
    
    t_inference_start = time.time()
    results = reader.readtext(image_np)
    t_inference_end = time.time()

    ocr_lines = []
    full_text_lines = []
    all_text = ""

    for bbox, text, confidence in results:
        if confidence < cfg.get("confidence_threshold", 0.3):
            continue
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        normalized_box = {
            "x": float(xmin / img_width),
            "y": float(ymin / img_height),
            "w": float((xmax - xmin) / img_width),
            "h": float((ymax - ymin) / img_height)
        }

        ocr_lines.append({
            "text": text,
            "confidence": float(confidence),
            "box": normalized_box
        })
        full_text_lines.append(text)
        all_text += " " + text
    
    t_parsing_start = time.time()
    
    # Track detected lines and bounding boxes
    ocr_lines = []
    full_text_lines = []
    all_text = ""
    
    for bbox, text, confidence in results:
        if confidence <= 0.5:
            continue
            
        # bbox layout: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        normalized_box = {
            "x": float(xmin / img_width),
            "y": float(ymin / img_height),
            "w": float((xmax - xmin) / img_width),
            "h": float((ymax - ymin) / img_height)
        }
        
        ocr_lines.append({
            "text": text,
            "confidence": float(confidence),
            "box": normalized_box
        })
        full_text_lines.append(text)
        all_text += " " + text
    
    # 1. Detect RAMQ
    ramq_pattern = r'\b([A-Z]{4})\s*(\d{4})\s*(\d{4})\b|\b([A-Z]{4})\s*(\d{8})\b'
    detected_ramq = None
    ramq_dob_info = None
    ramq_box = None
    
    # First search line-by-line to find corresponding box
    for line in ocr_lines:
        line_text = line["text"].upper()
        match = re.search(ramq_pattern, line_text)
        if match:
            cleaned = re.sub(r'[^A-Z0-9]', '', match.group(0))
            info = parse_ramq_dob_info(cleaned)
            if info:
                detected_ramq = f"{cleaned[:4]} {cleaned[4:8]} {cleaned[8:]}"
                ramq_dob_info = info
                ramq_box = line["box"]
                break
                
    # Fallback to global search if not found line-by-line
    if not detected_ramq:
        ramq_matches = re.finditer(ramq_pattern, all_text.upper())
        for match in ramq_matches:
            matched_str = match.group(0)
            cleaned = re.sub(r'[^A-Z0-9]', '', matched_str)
            info = parse_ramq_dob_info(cleaned)
            if info:
                detected_ramq = f"{cleaned[:4]} {cleaned[4:8]} {cleaned[8:]}"
                ramq_dob_info = info
                # Approximate box by finding first line containing initials
                for line in ocr_lines:
                    if cleaned[:4] in line["text"].upper():
                        ramq_box = line["box"]
                        break
                break
            
    # 2. Detect DOB
    dob_pattern = r'\b(19\d{2}|20\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b|\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](19\d{2}|20\d{2})\b'
    detected_dob = None
    dob_box = None
    
    # Search line-by-line first to get bbox
    for line in ocr_lines:
        match = re.search(dob_pattern, line["text"])
        if match:
            parts = [p for p in match.groups() if p]
            if len(parts) == 3:
                if len(parts[0]) == 4:
                    detected_dob = f"{parts[0]}-{parts[1]}-{parts[2]}"
                else:
                    detected_dob = f"{parts[2]}-{parts[1]}-{parts[0]}"
                dob_box = line["box"]
                break
                
    # Fallback to global regex search
    if not detected_dob:
        dob_matches = re.findall(dob_pattern, all_text)
        if dob_matches:
            for match in dob_matches:
                parts = [p for p in match if p]
                if len(parts) == 3:
                    if len(parts[0]) == 4:
                        detected_dob = f"{parts[0]}-{parts[1]}-{parts[2]}"
                    else:
                        detected_dob = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    # Locate line containing parts
                    for line in ocr_lines:
                        if parts[0] in line["text"] and parts[2] in line["text"]:
                            dob_box = line["box"]
                            break
                    break
                
    # Use RAMQ encoded DOB as validation/fallback
    if ramq_dob_info:
        ramq_dob = ramq_dob_info["dob"]
        if not detected_dob:
            detected_dob = ramq_dob
            dob_box = ramq_box
            
    # 3. Detect Name
    detected_name = None
    name_box = None
    candidate_last = None
    candidate_first = None
    last_box = None
    first_box = None

    # Keyword patterns across full text
    nom_pattern = r'\b(?:NOM|LAST\s*NAME|NOM\s*DE\s*FAMILLE)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)'
    prenom_pattern = r'\b(?:PRÉNOM|PRENOM|FIRST\s*NAME)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)'
    full_name_pattern = r'\b(?:NOM\s*COMPLET|FULL\s*NAME|PATIENT)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)\s*[,;:]?\s*([A-ZÀ-ÿ\'-]+)'

    # 1. Search line by line for explicit Nom and Prénom keywords
    for line in ocr_lines:
        text_up = line["text"].upper()
        
        # Check Nom
        if not candidate_last:
            match_nom = re.search(nom_pattern, text_up)
            if match_nom:
                candidate_last = match_nom.group(1)
                last_box = line["box"]

        # Check Prénom
        if not candidate_first:
            match_prenom = re.search(prenom_pattern, text_up)
            if match_prenom:
                candidate_first = match_prenom.group(1)
                first_box = line["box"]

        # Check Nom complet / Patient label
        match_full = re.search(full_name_pattern, text_up)
        if match_full and not (candidate_last and candidate_first):
            w1 = match_full.group(1)
            w2 = match_full.group(2)
            # Standard order in French medical forms: "NOM , Prénom"
            if not candidate_last:
                candidate_last = w1
                last_box = line["box"]
            if not candidate_first:
                candidate_first = w2
                first_box = line["box"]

    # 2. Validate against RAMQ initials if RAMQ is available
    if ramq_dob_info:
        initials = ramq_dob_info["initials"]
        last_init = initials[:3]
        first_init = initials[3]

        ramq_last = None
        ramq_first = None
        ramq_last_box = None
        ramq_first_box = None

        # Search across all OCR lines for RAMQ matching words
        for line in ocr_lines:
            words = re.findall(r'\b[a-zA-ZÀ-ÿ\'-]+\b', line["text"])
            for w in words:
                w_up = w.upper()
                if not ramq_last and w_up.startswith(last_init) and len(w) >= 3:
                    ramq_last = w
                    ramq_last_box = line["box"]
                elif not ramq_first and w_up.startswith(first_init) and len(w) >= 2:
                    ramq_first = w
                    ramq_first_box = line["box"]

        # If keyword-based names were found, check RAMQ initial match
        valid_keyword_match = True
        if candidate_last and not candidate_last.upper().startswith(last_init):
            valid_keyword_match = False
        if candidate_first and not candidate_first.upper().startswith(first_init):
            valid_keyword_match = False

        if valid_keyword_match and candidate_last and candidate_first:
            detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
        elif ramq_last and ramq_first:
            # Fallback to RAMQ matched words if keyword initials mismatched or missing
            candidate_last = ramq_last
            candidate_first = ramq_first
            last_box = ramq_last_box
            first_box = ramq_first_box
            detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
    
    # 3. If still not detected, fallback to whatever keyword candidate was extracted
    if not detected_name and candidate_last and candidate_first:
        detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
    elif not detected_name and candidate_last:
        detected_name = candidate_last.capitalize()
    
    # Calculate bounding box for name highlight
    if candidate_last or candidate_first:
        if last_box and first_box:
            xmin = min(last_box["x"], first_box["x"])
            ymin = min(last_box["y"], first_box["y"])
            xmax = max(last_box["x"] + last_box["w"], first_box["x"] + first_box["w"])
            ymax = max(last_box["y"] + last_box["h"], first_box["y"] + first_box["h"])
            name_box = {
                "x": xmin,
                "y": ymin,
                "w": xmax - xmin,
                "h": ymax - ymin
            }
        else:
            name_box = last_box or first_box
                
    t_parsing_end = time.time()
    t_end = time.time()
    
    model_inference_ms = round((t_inference_end - t_inference_start) * 1000, 2)
    regex_parsing_ms = round((t_parsing_end - t_parsing_start) * 1000, 2)
    total_latency_ms = round((t_end - t_start) * 1000, 2)
    
    success = bool(detected_name and detected_dob and detected_ramq)
    
    # Assemble log entry (without raw scanned_texts)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "success": success,
        "latency_ms": {
            "model_inference": model_inference_ms,
            "regex_parsing": regex_parsing_ms,
            "total": total_latency_ms
        },
        "detected_fields": {
            "name": detected_name,
            "dob": detected_dob,
            "ramq": detected_ramq
        },
        "highlights": {
            "name": name_box,
            "dob": dob_box,
            "ramq": ramq_box
        },
        "scanned_texts": full_text_lines
    }
    
    # Save structured logs
    log_scan_details(log_entry)
    
    return {
        "name": detected_name,
        "dob": detected_dob,
        "ramq": detected_ramq,
        "success": success,
        "highlights": {
            "name": name_box,
            "dob": dob_box,
            "ramq": ramq_box
        },
        "total_latency_ms": total_latency_ms
    }



# FOCUS HERE NOW
# Type of script #1 -  The actual code that transform image to RQMA-DB, etc.
    # Process logs - and latency logs - know the latency of each component of the process
    # Add more diverse simulated data
    # Real world tests  (Last - because more expensive)

# We'll FOCUS HERE LATER 
# Type of script #2 - how to generate synthetic data that is good
    # If possible to generate realistic data (images) - labels -  Millions of examples
    # Million of examples - means you can use AI (Need a lot of examples to generalize) ***

    
    

