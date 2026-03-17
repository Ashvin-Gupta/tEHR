"""
Tests for MIMIC and CPRD token translators.

Covers:
- translate(): each major token category returns sensible natural language
- is_measurable_concept(): correctly identifies measurable tokens
- is_numeric_value(): correctly identifies value tokens
- is_new_event_code(): correctly identifies event-boundary tokens
- is_header_concept(): correctly identifies demographic header tokens
- Edge cases: empty string, non-string, unknown token
"""
import pytest
from src.data.mimic_translator import MIMICTokenTranslator
from src.data.cprd_translator import CPRDTokenTranslator


# ===========================================================================
# MIMIC Token Translator
# ===========================================================================

class TestMIMICTokenTranslator:

    def test_lab_token_known(self, mimic_translator):
        result = mimic_translator.translate("LAB//51006")
        assert "Urea Nitrogen" in result
        assert result.endswith(";")

    def test_lab_token_unknown(self, mimic_translator):
        result = mimic_translator.translate("LAB//99999")
        # Falls back to the raw item-id string
        assert "99999" in result

    def test_diagnosis_icd10(self, mimic_translator):
        result = mimic_translator.translate("DIAGNOSIS//ICD//10//I10")
        assert "hypertension" in result.lower()
        assert result.endswith(";")

    def test_diagnosis_icd9(self, mimic_translator):
        result = mimic_translator.translate("DIAGNOSIS//ICD//9//4019")
        assert "hypertension" in result.lower()

    def test_procedure_token(self, mimic_translator):
        result = mimic_translator.translate("PROCEDURE//ICD//10//0BH17EZ")
        assert len(result) > 2

    def test_time_token(self, mimic_translator):
        result = mimic_translator.translate("TIME 2020-01")
        assert "2020-01" in result
        assert result.endswith(";")

    def test_medication_start(self, mimic_translator):
        result = mimic_translator.translate("MEDICATION//START//metformin")
        assert "metformin" in result.lower()
        assert "start" in result.lower()

    def test_medication_other(self, mimic_translator):
        result = mimic_translator.translate("MEDICATION//aspirin//100mg")
        assert "aspirin" in result.lower()

    def test_admission_token(self, mimic_translator):
        result = mimic_translator.translate("ADMISSION//EMERGENCY")
        assert result.endswith(";")
        assert "admission" in result.lower()

    def test_gender_female(self, mimic_translator):
        assert mimic_translator.translate("GENDER//F") == " Female;"

    def test_gender_male(self, mimic_translator):
        assert mimic_translator.translate("GENDER//M") == " Male;"

    def test_race_token(self, mimic_translator):
        result = mimic_translator.translate("RACE//RACE_WHITE")
        assert "White" in result or "white" in result.lower()

    def test_start_token(self, mimic_translator):
        result = mimic_translator.translate("<start>")
        assert "<start>" in result

    def test_numeric_passthrough(self, mimic_translator):
        result = mimic_translator.translate("13.5")
        assert "13.5" in result

    def test_empty_string(self, mimic_translator):
        result = mimic_translator.translate("")
        assert result == ""

    def test_non_string_input(self, mimic_translator):
        result = mimic_translator.translate(None)
        assert result == ""

    # --- predicate methods ---

    def test_is_measurable_lab(self, mimic_translator):
        assert mimic_translator.is_measurable_concept("LAB//51006") is True

    def test_is_measurable_age(self, mimic_translator):
        assert mimic_translator.is_measurable_concept("AGE") is True

    def test_is_measurable_bmi(self, mimic_translator):
        assert mimic_translator.is_measurable_concept("BMI") is True

    def test_is_not_measurable_diagnosis(self, mimic_translator):
        assert mimic_translator.is_measurable_concept("DIAGNOSIS//ICD//10//I10") is False

    def test_is_numeric_value_integer(self, mimic_translator):
        assert mimic_translator.is_numeric_value("55") is True

    def test_is_numeric_value_float(self, mimic_translator):
        assert mimic_translator.is_numeric_value("13.5") is True

    def test_is_numeric_value_text(self, mimic_translator):
        # Text token with no digits is NOT a value
        assert mimic_translator.is_numeric_value("Haemoglobin") is False

    def test_is_new_event_lab(self, mimic_translator):
        assert mimic_translator.is_new_event_code("LAB//51006") is True

    def test_is_new_event_diagnosis(self, mimic_translator):
        assert mimic_translator.is_new_event_code("DIAGNOSIS//ICD//10//I10") is True

    def test_is_not_new_event_number(self, mimic_translator):
        assert mimic_translator.is_new_event_code("13.5") is False

    def test_is_header_start(self, mimic_translator):
        assert mimic_translator.is_header_concept("<start>") is True

    def test_is_header_gender(self, mimic_translator):
        assert mimic_translator.is_header_concept("GENDER//F") is True

    def test_is_header_age(self, mimic_translator):
        assert mimic_translator.is_header_concept("AGE") is True

    def test_is_not_header_lab(self, mimic_translator):
        assert mimic_translator.is_header_concept("LAB//51006") is False

    def test_is_not_header_diagnosis(self, mimic_translator):
        assert mimic_translator.is_header_concept("DIAGNOSIS//ICD//10//I10") is False


# ===========================================================================
# CPRD Token Translator
# ===========================================================================

class TestCPRDTokenTranslator:

    def test_instantiation(self, cprd_translator):
        """
        CPRDTokenTranslator must be instantiable.

        KNOWN BUG: CPRDTokenTranslator is missing the abstract method
        `is_numeric_value` required by BaseTokenTranslator.
        This test will raise TypeError until the method is added:

            def is_numeric_value(self, token: str) -> bool:
                s = str(token)
                if s.replace(".", "", 1).isdigit():
                    return True
                # Quantile bins Q1..Q9
                if s.startswith("Q") and len(s) <= 4 and s[1:].isdigit():
                    return True
                return False
        """
        assert cprd_translator is not None

    def test_medical_token_known(self, cprd_translator):
        result = cprd_translator.translate("MEDICAL//ABCD")
        assert "Haemoglobin" in result
        assert result.endswith("; ")

    def test_medical_token_unknown(self, cprd_translator):
        result = cprd_translator.translate("MEDICAL//ZZZZ")
        assert len(result) > 2  # falls back to code

    def test_lab_token_known(self, cprd_translator):
        result = cprd_translator.translate("LAB//HAB1")
        assert "HbA1c" in result

    def test_gender_token(self, cprd_translator):
        result = cprd_translator.translate("GENDER//Female")
        assert "Gender" in result or "gender" in result.lower()

    def test_region_token(self, cprd_translator):
        result = cprd_translator.translate("REGION//1")
        assert "North East" in result

    def test_age_token(self, cprd_translator):
        result = cprd_translator.translate("AGE")
        assert "AGE" in result

    def test_numeric_float(self, cprd_translator):
        result = cprd_translator.translate("3.5")
        assert "3.5" in result

    def test_quantile_value(self, cprd_translator):
        result = cprd_translator.translate("Q3")
        assert "3" in result  # quantile bin renders as the number

    def test_time_interval_token(self, cprd_translator):
        result = cprd_translator.translate("<time_interval_1M>")
        assert "month" in result.lower()

    def test_start_token(self, cprd_translator):
        result = cprd_translator.translate("<start>")
        assert "<start>" in result

    def test_empty_string(self, cprd_translator):
        result = cprd_translator.translate("")
        assert result == "Unknown; "

    def test_non_string_input(self, cprd_translator):
        result = cprd_translator.translate(None)
        assert result == ""

    # --- predicate methods ---

    def test_is_measurable_lab(self, cprd_translator):
        assert cprd_translator.is_measurable_concept("LAB//HAB1") is True

    def test_is_measurable_measurement(self, cprd_translator):
        assert cprd_translator.is_measurable_concept("MEASUREMENT//XM0LY") is True

    def test_is_measurable_bmi(self, cprd_translator):
        assert cprd_translator.is_measurable_concept("MEDICAL//BMI") is True

    def test_is_not_measurable_medical(self, cprd_translator):
        assert cprd_translator.is_measurable_concept("MEDICAL//ABCD") is False

    def test_is_numeric_float(self, cprd_translator):
        assert cprd_translator.is_numeric_value("13.5") is True

    def test_is_numeric_integer(self, cprd_translator):
        assert cprd_translator.is_numeric_value("55") is True

    def test_is_numeric_quantile(self, cprd_translator):
        assert cprd_translator.is_numeric_value("Q2") is True

    def test_is_not_numeric_text(self, cprd_translator):
        assert cprd_translator.is_numeric_value("MEDICAL//ABCD") is False

    def test_is_header_start(self, cprd_translator):
        assert cprd_translator.is_header_concept("<start>") is True

    def test_is_header_age(self, cprd_translator):
        assert cprd_translator.is_header_concept("AGE") is True

    def test_is_header_gender(self, cprd_translator):
        assert cprd_translator.is_header_concept("GENDER//F") is True

    def test_is_not_header_lab(self, cprd_translator):
        assert cprd_translator.is_header_concept("LAB//HAB1") is False

    def test_is_not_header_medical(self, cprd_translator):
        assert cprd_translator.is_header_concept("MEDICAL//ABCD") is False

    def test_is_new_event_medical(self, cprd_translator):
        assert cprd_translator.is_new_event_code("MEDICAL//ABCD") is True

    def test_is_new_event_lab(self, cprd_translator):
        assert cprd_translator.is_new_event_code("LAB//HAB1") is True

    def test_is_not_new_event_number(self, cprd_translator):
        assert cprd_translator.is_new_event_code("13.5") is False
