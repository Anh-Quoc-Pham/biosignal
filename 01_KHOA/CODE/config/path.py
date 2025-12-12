from pathlib import Path
# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "00_DATA" / "01_RAW"
STRUCTURED_ROOT = PROJECT_ROOT / "01_KHOA" / "DATA"/ "STRUCTURED"
PREPROCESSED_ROOT = PROJECT_ROOT / "01_KHOA" / "DATA"/ "PREPROCESSED"

# Sampling rate
FS = 500  # Hz
FS_IMU = 50  # IMU sampling rate