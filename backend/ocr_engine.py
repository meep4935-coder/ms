import re
import datetime
import time
import json
import os
import easyocr
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from pathlib import Path
from logger import write_scan_log

# Helper function to check CUDA GPU availability for EasyOCR
def get_device_gpu_status():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

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
    try:
        mm_val = int(clean_ramq[6:8])
        dd_str = clean_ramq[8:10]
        yy = int(yy_str)
    except ValueError:
        return None
    
    # Check gender
    is_female = False
    if mm_val > 50:
        is_female = True
        mm_val -= 50
        
    if not (1 <= mm_val <= 12):
        return None
        
    # Determine full year
    current_year = datetime.datetime.now().year
    current_yy = current_year % 100
    
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

def compute_blur_score(gray_img: np.ndarray) -> float:
    """Computes Laplacian variance as a proxy for image sharpness."""
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def estimate_rotation_angle(gray_img: np.ndarray) -> float:
    """Estimates skew angle of text lines using Hough Line Transform."""
    try:
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
            if -45 < angle < 45:
                angles.append(angle)
        if len(angles) == 0:
            return 0.0
        return float(np.median(angles))
    except Exception:
        return 0.0

def preprocess_image(image_np: np.ndarray, file_size_bytes: int):
    """
    Applies quality analysis and preprocessing:
    - Blur score (Laplacian variance)
    - Quality classification (sharp >= 300, soft >= 100, blurry < 100)
    - Skew correction if needed
    - CLAHE contrast enhancement
    - Denoising if soft/blurry
    - Upscaling if image resolution is too small
    """
    t_start = time.time()
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    blur_score = compute_blur_score(gray)
    if blur_score >= 300:
        quality_class = "sharp"
    elif blur_score >= 100:
        quality_class = "soft"
    else:
        quality_class = "blurry"

    rotation_angle = estimate_rotation_angle(gray)
    contrast_before = float(gray.std())
    
    steps_applied = []
    processed = image_np.copy()
    
    # 1. Rotate if significant skew detected
    if abs(rotation_angle) > 1.5:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        processed = cv2.warpAffine(processed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        steps_applied.append(f"deskew_{rotation_angle:.1f}deg")
    
    # 2. Upscale if low resolution
    upscaled = False
    if w < 1200 or h < 1200:
        scale = 1.5
        processed = cv2.resize(processed, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        steps_applied.append(f"upscale_{scale}x")
        upscaled = True

    # 3. CLAHE Contrast Enhancement
    lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    steps_applied.append("clahe_contrast")
    
    # 4. Denoise if blurry or soft
    if quality_class in ("blurry", "soft"):
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 7, 7, 7, 21)
        steps_applied.append("denoise_nlmeans")
        
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    contrast_after = float(proc_gray.std())
    
    t_end = time.time()
    
    quality_metrics = {
        "file_size_bytes": file_size_bytes,
        "resolution": [w, h],
        "blur_score": round(blur_score, 2),
        "quality_class": quality_class,
        "rotation_angle_deg": round(rotation_angle, 2)
    }
    
    preprocess_info = {
        "steps_applied": steps_applied,
        "contrast_before": round(contrast_before, 2),
        "contrast_after": round(contrast_after, 2),
        "upscaled": upscaled,
        "latency_ms": round((t_end - t_start) * 1000, 2)
    }
    
    return processed, quality_metrics, preprocess_info

def extract_patient_info(image_bytes: bytes, reader: easyocr.Reader = None):
    """
    Processes image bytes, runs preprocessing, executes OCR (with multi-scale fallback if needed),
    parses Quebec patient fields (RAMQ, DOB, Name), and generates an exhaustive step-by-step trace log.
    """
    t_start = time.time()
    trace_events = []
    
    def add_trace(event: str, detail: str):
        t_elapsed = round((time.time() - t_start) * 1000, 2)
        trace_events.append({"event": event, "t_ms": t_elapsed, "detail": detail})
        
    add_trace("IMAGE_LOAD", f"Loaded raw image bytes ({len(image_bytes)} bytes)")
    
    # Load PIL image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    raw_width, raw_height = image.size
    image_np = np.array(image)
    
    # Run preprocessing & quality analysis
    add_trace("PREPROCESS_START", "Analyzing quality, blur, and applying enhancement pipeline")
    processed_np, quality_metrics, preprocess_info = preprocess_image(image_np, len(image_bytes))
    add_trace("PREPROCESS_END", f"Quality: {quality_metrics['quality_class']} (blur_score={quality_metrics['blur_score']}), steps={preprocess_info['steps_applied']}")
    
    # Initialize easyocr reader if not provided
    if reader is None:
        add_trace("MODEL_INIT_FALLBACK", "No singleton reader provided, initializing temporary EasyOCR instance")
        use_gpu = get_device_gpu_status()
        reader = easyocr.Reader(['fr', 'en'], gpu=use_gpu, verbose=False)
        
    # Dynamic confidence thresholding: if blurry, lower threshold to capture soft text
    base_threshold = 0.3
    if quality_metrics["quality_class"] == "blurry":
        base_threshold = 0.15
        add_trace("DYNAMIC_THRESHOLD", "Blurry image detected; lowering OCR confidence threshold to 0.15")
    else:
        add_trace("DYNAMIC_THRESHOLD", f"Confidence threshold set to {base_threshold}")
        
    # Run primary OCR pass
    t_ocr_start = time.time()
    add_trace("OCR_START", f"Running primary EasyOCR pass on preprocessed image")
    raw_results = reader.readtext(processed_np)
    t_ocr_end = time.time()
    ocr_inference_ms = round((t_ocr_end - t_ocr_start) * 1000, 2)
    
    # Check if multi-scale pass is needed
    multiscale_fallback = False
    scale_factor = 1.0
    if len(raw_results) < 3 or (len(raw_results) > 0 and np.mean([r[2] for r in raw_results]) < 0.4):
        add_trace("MULTISCALE_FALLBACK", "Primary OCR returned sparse/low-confidence results. Running 1.5x multi-scale pass.")
        multiscale_fallback = True
        scale_factor = 1.5
        h_p, w_p = processed_np.shape[:2]
        upscaled_np = cv2.resize(processed_np, (int(w_p * 1.5), int(h_p * 1.5)), interpolation=cv2.INTER_CUBIC)
        t_ms_start = time.time()
        second_results = reader.readtext(upscaled_np)
        t_ms_end = time.time()
        ocr_inference_ms += round((t_ms_end - t_ms_start) * 1000, 2)
        
        # Scale bounding boxes back to processed_np coordinates
        scaled_back = []
        for bbox, text, conf in second_results:
            box_scaled = [[p[0] / 1.5, p[1] / 1.5] for p in bbox]
            scaled_back.append((box_scaled, text, conf))
            
        # Combine and deduplicate
        existing_texts = set(r[1].strip().upper() for r in raw_results)
        for item in scaled_back:
            if item[1].strip().upper() not in existing_texts:
                raw_results.append(item)
                
    proc_height, proc_width = processed_np.shape[:2]
    
    # Process OCR results (SINGLE PASS FIX)
    ocr_lines = []
    full_text_lines = []
    raw_ocr_dump = []
    all_text = ""
    
    above_threshold_count = 0
    total_conf = 0.0
    
    for bbox, text, confidence in raw_results:
        conf_float = float(confidence)
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        normalized_box = {
            "x": float(xmin / proc_width),
            "y": float(ymin / proc_height),
            "w": float((xmax - xmin) / proc_width),
            "h": float((ymax - ymin) / proc_height)
        }
        
        raw_ocr_dump.append({
            "text": text,
            "confidence": conf_float,
            "box": normalized_box,
            "passed_threshold": conf_float >= base_threshold
        })
        
        if conf_float >= base_threshold:
            above_threshold_count += 1
            total_conf += conf_float
            ocr_lines.append({
                "text": text,
                "confidence": conf_float,
                "box": normalized_box
            })
            full_text_lines.append(text)
            all_text += " " + text
            
    avg_confidence = round(total_conf / above_threshold_count, 4) if above_threshold_count > 0 else 0.0
    add_trace("OCR_END", f"Found {len(raw_results)} total regions, {above_threshold_count} above threshold {base_threshold}. Avg confidence: {avg_confidence}")

    # Regex Extraction Phase
    t_parsing_start = time.time()
    add_trace("REGEX_PARSING_START", "Searching for RAMQ, DOB, and Name patterns")
    
    # 1. Detect RAMQ
    ramq_pattern = r'\b([A-Z]{4})\s*(\d{4})\s*(\d{4})\b|\b([A-Z]{4})\s*(\d{8})\b'
    detected_ramq = None
    ramq_dob_info = None
    ramq_box = None
    ramq_extraction_details = {"found": False, "method": None, "matched_text": None}

    # Line-by-line RAMQ search
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
                ramq_extraction_details = {
                    "found": True,
                    "method": "line_by_line",
                    "matched_text": line["text"],
                    "cleaned": cleaned
                }
                add_trace("REGEX_RAMQ", f"Found RAMQ line-by-line: '{detected_ramq}'")
                break
                
    # Global RAMQ search fallback
    if not detected_ramq:
        ramq_matches = re.finditer(ramq_pattern, all_text.upper())
        for match in ramq_matches:
            matched_str = match.group(0)
            cleaned = re.sub(r'[^A-Z0-9]', '', matched_str)
            info = parse_ramq_dob_info(cleaned)
            if info:
                detected_ramq = f"{cleaned[:4]} {cleaned[4:8]} {cleaned[8:]}"
                ramq_dob_info = info
                for line in ocr_lines:
                    if cleaned[:4] in line["text"].upper():
                        ramq_box = line["box"]
                        break
                ramq_extraction_details = {
                    "found": True,
                    "method": "global_regex_fallback",
                    "matched_text": matched_str,
                    "cleaned": cleaned
                }
                add_trace("REGEX_RAMQ", f"Found RAMQ via global search: '{detected_ramq}'")
                break

    if not detected_ramq:
        add_trace("REGEX_RAMQ_FAILED", "No valid 12-char Quebec RAMQ pattern matched")

    # 2. Detect DOB
    dob_pattern = r'\b(19\d{2}|20\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b|\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](19\d{2}|20\d{2})\b'
    detected_dob = None
    dob_box = None
    dob_extraction_details = {"found": False, "method": None, "source": None}

    # Line-by-line DOB search
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
                dob_extraction_details = {
                    "found": True,
                    "method": "explicit_line_regex",
                    "matched_text": line["text"]
                }
                add_trace("REGEX_DOB", f"Found explicit DOB in text: '{detected_dob}'")
                break

    # Global DOB search fallback
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
                    for line in ocr_lines:
                        if parts[0] in line["text"] and parts[2] in line["text"]:
                            dob_box = line["box"]
                            break
                    dob_extraction_details = {
                        "found": True,
                        "method": "global_regex_fallback",
                        "matched_text": str(match)
                    }
                    add_trace("REGEX_DOB", f"Found DOB via global search: '{detected_dob}'")
                    break

    # RAMQ encoded DOB fallback/validation
    if ramq_dob_info:
        ramq_dob = ramq_dob_info["dob"]
        if not detected_dob:
            detected_dob = ramq_dob
            dob_box = ramq_box
            dob_extraction_details = {
                "found": True,
                "method": "ramq_checksum_decoded",
                "source": "RAMQ number"
            }
            add_trace("REGEX_DOB_RAMQ_FALLBACK", f"Derived DOB from RAMQ checksum: '{detected_dob}'")
        elif detected_dob == ramq_dob:
            add_trace("REGEX_DOB_VALIDATED", f"Explicit DOB matches RAMQ decoded DOB ({detected_dob})")
        else:
            add_trace("REGEX_DOB_MISMATCH_WARN", f"Explicit DOB ({detected_dob}) differs from RAMQ DOB ({ramq_dob})")

    if not detected_dob:
        add_trace("REGEX_DOB_FAILED", "No DOB pattern matched and RAMQ checksum unavailable")

    # 3. Detect Name
    detected_name = None
    name_box = None
    candidate_last = None
    candidate_first = None
    last_box = None
    first_box = None
    name_extraction_details = {"found": False, "method": None}

    nom_pattern = r'\b(?:NOM|LAST\s*NAME|NOM\s*DE\s*FAMILLE)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)'
    prenom_pattern = r'\b(?:PRÉNOM|PRENOM|FIRST\s*NAME)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)'
    full_name_pattern = r'\b(?:NOM\s*COMPLET|FULL\s*NAME|PATIENT)\s*[:\.-]?\s*([A-ZÀ-ÿ\'-]+)\s*[,;:]?\s*([A-ZÀ-ÿ\'-]+)'

    for line in ocr_lines:
        text_up = line["text"].upper()
        if not candidate_last:
            match_nom = re.search(nom_pattern, text_up)
            if match_nom:
                candidate_last = match_nom.group(1)
                last_box = line["box"]

        if not candidate_first:
            match_prenom = re.search(prenom_pattern, text_up)
            if match_prenom:
                candidate_first = match_prenom.group(1)
                first_box = line["box"]

        match_full = re.search(full_name_pattern, text_up)
        if match_full and not (candidate_last and candidate_first):
            w1 = match_full.group(1)
            w2 = match_full.group(2)
            if not candidate_last:
                candidate_last = w1
                last_box = line["box"]
            if not candidate_first:
                candidate_first = w2
                first_box = line["box"]

    # Validate against RAMQ initials
    if ramq_dob_info:
        initials = ramq_dob_info["initials"]
        last_init = initials[:3]
        first_init = initials[3]

        ramq_last = None
        ramq_first = None
        ramq_last_box = None
        ramq_first_box = None

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

        valid_keyword_match = True
        if candidate_last and not candidate_last.upper().startswith(last_init):
            valid_keyword_match = False
        if candidate_first and not candidate_first.upper().startswith(first_init):
            valid_keyword_match = False

        if valid_keyword_match and candidate_last and candidate_first:
            detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
            name_extraction_details = {"found": True, "method": "keywords_ramq_validated"}
            add_trace("REGEX_NAME", f"Extracted name via keywords (RAMQ initials validated): '{detected_name}'")
        elif ramq_last and ramq_first:
            candidate_last = ramq_last
            candidate_first = ramq_first
            last_box = ramq_last_box
            first_box = ramq_first_box
            detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
            name_extraction_details = {"found": True, "method": "ramq_initials_word_search"}
            add_trace("REGEX_NAME", f"Extracted name via RAMQ initials search: '{detected_name}'")

    if not detected_name and candidate_last and candidate_first:
        detected_name = f"{candidate_last.capitalize()}, {candidate_first.capitalize()}"
        name_extraction_details = {"found": True, "method": "unvalidated_keywords"}
        add_trace("REGEX_NAME", f"Extracted name via unvalidated keywords: '{detected_name}'")
    elif not detected_name and candidate_last:
        detected_name = candidate_last.capitalize()
        name_extraction_details = {"found": True, "method": "last_name_only"}
        add_trace("REGEX_NAME", f"Extracted last name only: '{detected_name}'")

    if not detected_name:
        add_trace("REGEX_NAME_FAILED", "Could not locate patient name")

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
    
    regex_parsing_ms = round((t_parsing_end - t_parsing_start) * 1000, 2)
    total_latency_ms = round((t_end - t_start) * 1000, 2)
    
    success = bool(detected_name and detected_dob and detected_ramq)
    
    # Determine failure reason if unsuccessful
    failure_reason = None
    if not success:
        if len(raw_results) == 0:
            failure_reason = "NO_TEXT_DETECTED"
        elif above_threshold_count == 0:
            failure_reason = "LOW_CONFIDENCE_ALL_FILTERED"
        elif not detected_ramq:
            failure_reason = "PATTERN_NO_MATCH_RAMQ"
        elif not detected_dob:
            failure_reason = "PATTERN_NO_MATCH_DOB"
        elif not detected_name:
            failure_reason = "NAME_NOT_EXTRACTABLE"
            
    add_trace("RESULT", f"Success={success}, FailureReason={failure_reason}, Latency={total_latency_ms}ms")
    
    # Comprehensive per-scan log entry
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "success": success,
        "failure_reason": failure_reason,
        "image_quality": quality_metrics,
        "preprocessing": preprocess_info,
        "ocr": {
            "confidence_threshold_used": base_threshold,
            "scale_used": scale_factor,
            "multiscale_fallback": multiscale_fallback,
            "regions_total": len(raw_results),
            "regions_above_threshold": above_threshold_count,
            "avg_confidence": avg_confidence,
            "inference_ms": ocr_inference_ms,
            "raw_results": raw_ocr_dump
        },
        "extraction": {
            "ramq": ramq_extraction_details,
            "dob": dob_extraction_details,
            "name": name_extraction_details
        },
        "latency_ms": {
            "image_load": trace_events[0]["t_ms"],
            "preprocessing": preprocess_info["latency_ms"],
            "ocr_inference": ocr_inference_ms,
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
        "scanned_texts": full_text_lines,
        "pipeline_trace": trace_events
    }
    
    # Write to files (logs/scan_*.json and logs/pipeline.jsonl)
    filepath = write_scan_log(log_entry)
    print(f"[OCR Engine] Completed scan. Success: {success}. Log saved to: {filepath}")
    
    # Return API response (lean response + diagnostics object)
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
        "total_latency_ms": total_latency_ms,
        "diagnostics": {
            "failure_reason": failure_reason,
            "image_quality": quality_metrics,
            "preprocessing": preprocess_info,
            "ocr_summary": {
                "regions_total": len(raw_results),
                "regions_above_threshold": above_threshold_count,
                "avg_confidence": avg_confidence,
                "confidence_threshold_used": base_threshold,
                "multiscale_fallback": multiscale_fallback
            },
            "latency_breakdown": log_entry["latency_ms"],
            "pipeline_trace": trace_events
        }
    }
