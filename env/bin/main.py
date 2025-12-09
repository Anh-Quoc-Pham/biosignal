import os
import glob
import io

import numpy as np
import pandas as pd

# === PATHS ===
PROJECT_ROOT   = "/"
DATA_ROOT      = "00_DATA/01_RAW"
STRUCTURED_ROOT = "00_DATA/02_STUCTURE_DATA"

# Sampling rate
FS = 500  # Hz

# Device naming
DEVICE_ROLES = {
    1: "EEG_1",
    2: "EMG_2",
    3: "EMG_3",
}

IMU_ROLES = {
    1: "IMU_1",
    2: "IMU_2",
    3: "IMU_3",
}

print("PROJECT_ROOT  :", PROJECT_ROOT)
print("DATA_ROOT     :", DATA_ROOT)
print("STRUCTURED_ROOT:", STRUCTURED_ROOT)
print("\nDevice roles:")
for d, name in DEVICE_ROLES.items():
    print(f"  Device {d} → {name}")
for d, name in IMU_ROLES.items():
    print(f"  Device {d} → {name}")
