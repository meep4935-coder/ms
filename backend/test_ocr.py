# Simple test script to verify RAMQ & DOB extraction regex rules.
from ocr_engine import parse_ramq_dob_info

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
    
    print("All backend local unit tests passed successfully!")

if __name__ == "__main__":
    test_ramq_parsing()
