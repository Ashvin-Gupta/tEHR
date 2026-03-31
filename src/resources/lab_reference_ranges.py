"""Reference ranges for lab and vital terms used in clustering analysis."""

LAB_REFERENCE_RANGES = {
    "Lab MCV": (80.0, 100.0),             # fL
    "Lab CRP": (0.0, 5.0),                # mg/L (Anything > 5 is clinically elevated)
    "Lab Hemoglobin": (120.0, 160.0),     # g/L (Compromise range; females 120-150, males 130-170)
    "Lab TIBC": (45.0, 85.0),             # umol/L
    "Lab HbA1c": (20.0, 42.0),            # mmol/mol (<42 is normal, 42-47 pre-diabetic, >48 diabetic)
    "Lab Plasma viscosity": (1.50, 1.72), # mPa.s
    "Lab ESR": (0.0, 20.0),               # mm/hr (Highly age/sex dependent, but 20 is a safe general upper limit)
    "Lab GGT": (0.0, 50.0),               # U/L
    "Lab Lymphocytes": (1.0, 4.0),        # 10^9/L
    "Lab Platelets": (150.0, 400.0),      # 10^9/L
    "Lab AST": (0.0, 40.0),               # U/L
    "Lab ALP": (30.0, 130.0),             # U/L
    "Lab Ferritin": (15.0, 300.0),        # ug/L
    "Lab MCH": (27.0, 32.0),              # pg
    "Lab Calcium serum": (2.20, 2.60),    # mmol/L
    "Lab Neutrophils": (2.0, 7.5),        # 10^9/L

    # Fasting glucose normally 4.0-5.9. Random glucose normal < 7.8.
    "Lab Glucose": (4.0, 5.9),            # mmol/L

    # Assuming "Cholesterol triglycerides" refers specifically to triglycerides
    "Lab Cholesterol triglycerides": (0.0, 1.7), # mmol/L (Fasting)

    "Lab Bilirubin": (0.0, 21.0),         # umol/L
    "Lab Anti-ttg": (0.0, 14.9),          # U/mL (Assay dependent, usually <15 is negative)
    "Lab Plasma proteins": (60.0, 80.0),  # g/L (Total protein)
    "Lab Amylase": (0.0, 100.0),          # U/L
    "Lab ALT": (0.0, 40.0),               # U/L
    "Lab Urea serum": (2.5, 7.8),         # mmol/L
    "Lab CA125": (0.0, 35.0),             # U/mL
    "Lab Creatinine serum": (60.0, 110.0),# umol/L (Compromise; females 60-90, males 70-110)
    "Lab Albumin serum": (35.0, 50.0),    # g/L
    "Lab WBC": (4.0, 11.0),               # 10^9/L
    "Lab Creatinine urine": (2.0, 20.0),  # mmol/L (For a random sample; highly variable)
    "Lab Iron": (10.0, 30.0),             # umol/L (Serum iron)

    # --- SPECIAL CASES ---
    # Blood pressure is usually recorded as two separate features (Systolic and Diastolic).
    # If your dataset has averaged them into Mean Arterial Pressure (MAP), normal is ~ 70-100.
    # If it is tracking Systolic only, normal is 90-120. I've set it to Systolic defaults:
    "Lab Blood pressure": (90.0, 120.0),  # mmHg

    # H. Pylori is usually a qualitative positive/negative (1 or 0) rather than a continuous lab.
    # If your model outputs a probability or titer, < 0.9 is typically "negative".
    "Lab H_pylori": (0.0, 0.99)
}
