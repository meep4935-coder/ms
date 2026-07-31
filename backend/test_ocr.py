# Simple test script to verify RAMQ & DOB extraction regex rules and preprocessing pipeline.
import numpy as np
import cv2
from ocr_engine import parse_ramq_dob_info, compute_blur_score, preprocess_image

def test_ramq_parsing():
    # RAMQ structure: AAAA YYMMDD XX
    # Marie Tremblay: TREM, female, born 1985-06-15 -> TREM 85 (06+50=56) 15 -> TREM 8556 1512
    res_female = parse_ramq_dob_info("TREM 8556 1512")
    assert res_female is not None
    assert res_female["dob"] == "1985-06-15"
    assert res_female["gender"] == "F"
    assert res_female["initials"] == "TREM"
    
    # Jean Tremblay: TREM, male, born 1992-12-04 -> TREM 92 12 04 -> TREM 9212 0434
    res_male = parse_ramq_dob_info("TREM 9212 0434")
    assert res_male is not None
    assert res_male["dob"] == "1992-12-04"
    assert res_male["gender"] == "M"
    
    # Test 2000s born patient: born 2005-01-20
    res_2000 = parse_ramq_dob_info("TREM 0501 2043")
    assert res_2000 is not None
    assert res_2000["dob"] == "2005-01-20"
    
    print("RAMQ regex parsing tests passed!")

def test_image_preprocessing():
    # Synthetic clean image
    clean_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.putText(clean_img, "RAMQ TEST 1234", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
    blur_score = compute_blur_score(gray)
    assert blur_score > 0
    
    # Run preprocess
    processed_np, quality_metrics, preprocess_info = preprocess_image(clean_img, file_size_bytes=1024)
    assert quality_metrics["quality_class"] in ("sharp", "soft", "blurry")
    assert "resolution" in quality_metrics
    assert len(preprocess_info["steps_applied"]) > 0
    print("Image preprocessing unit tests passed!")

if __name__ == "__main__":
    test_ramq_parsing()
    test_image_preprocessing()
    print("All backend local unit tests passed successfully!")
