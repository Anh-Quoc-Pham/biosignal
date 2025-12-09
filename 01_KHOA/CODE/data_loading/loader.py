import os
import glob
import io
from config.path import PROJECT_ROOT, STRUCTURED_ROOT, DATA_ROOT

import pandas as pd

# Sampling rate
FS = 500  # Hz
FS_IMU = 50  # IMU sampling rate

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
print("STRUCTURED_ROOT:", STRUCTURED_ROOT)
print("\nDevice roles:")

for d, name in DEVICE_ROLES.items():
    print(f"  Device {d} → {name}")
for d, name in IMU_ROLES.items():
    print(f"  Device {d} → {name}")

# Utility: List Subjects

def list_subjects(data_root: str):
    """Return sorted list of subject IDs, e.g. ['S01','S02','S04']."""
    subjects = [
        d for d in os.listdir(data_root)
        if d.startswith("S") and os.path.isdir(os.path.join(data_root, d))
    ]
    return sorted(subjects)

subjects = list_subjects(DATA_ROOT)
print("Detected subjects:", subjects)

def get_subject_paths(subject_id: str):
    """Return subject root, raw folder, and trigger folder."""
    subj_root = os.path.join(DATA_ROOT, subject_id)
    raw_dir   = os.path.join(subj_root, "raw")
    trig_dir  = os.path.join(subj_root, "triggers")
    return subj_root, raw_dir, trig_dir

def list_trials_for_subject(subject_id: str):
    """
    List raw CSV files and trigger CSV files for a subject.
    raw:      Sxx_T*.csv
    triggers: Sxx_T*triggers*.csv
    """
    _, raw_dir, trig_dir = get_subject_paths(subject_id)

    raw_files = sorted(glob.glob(os.path.join(raw_dir,  f"{subject_id}_T*.csv")))
    trig_files = sorted(glob.glob(os.path.join(trig_dir, f"{subject_id}_T*triggers*.csv")))

    return raw_files, trig_files
#==============================

# Clean header we expect in MindRove CSV (simple format)

#===============================
HEADER_SIMPLE = [
    "Device number",
    "Channel1", "Channel2", "Channel3", "Channel4",
    "Channel5", "Channel6", "Channel7", "Channel8",
    "GyroX", "GyroY", "GyroZ",
    "AccX", "AccY", "AccZ",
    "Timestamp",
]

def load_mindrove_csv_simple(path: str) -> pd.DataFrame:
    """
    Load a MindRove CSV and ensure a clean header:

        Device number, Channel1..Channel8, GyroX/Y/Z, AccX/Y/Z, Timestamp

    - Skips initial 'Device ...' metadata lines.
    - Forces the header to HEADER_SIMPLE.
    - Parses as tab-separated ('\\t').
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # Skip 'Device ...' lines
    start = 0
    while start < len(lines) and lines[start].startswith("Device "):
        start += 1

    if start >= len(lines):
        raise ValueError(f"No header line found in file: {path}")

    # Replace header line with our clean header
    lines[start] = "\t".join(HEADER_SIMPLE)

    text = "\n".join(lines[start:])

    # Read as tab-separated
    df = pd.read_csv(io.StringIO(text), sep="\t")

    # Enforce column names if lengths match
    if len(df.columns) == len(HEADER_SIMPLE):
        df.columns = HEADER_SIMPLE

    # Convert numeric columns
    for col in HEADER_SIMPLE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fully empty rows
    df = df.dropna(how="all")

    return df

EXG_COLS = [f"Channel{i}" for i in range(1, 9)]
IMU_COLS = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"]

def split_devices(df_raw: pd.DataFrame):
    """
    Split MindRove mixed CSV into 6 logical streams:

      EEG_1: Device 1, Timestamp + Channel1–Channel6
      EMG_2: Device 2, Timestamp + Channel1–Channel8
      EMG_3: Device 3, Timestamp + Channel1–Channel8
      IMU_1: Device 1, Timestamp + GyroX/Y/Z + AccX/Y/Z
      IMU_2: Device 2, same IMU cols
      IMU_3: Device 3, same IMU cols

    Returns:
      dict: { 'EEG_1': df, 'EMG_2': df, ..., 'IMU_3': df }
    """
    streams = {}

    for dev_num, role_name in DEVICE_ROLES.items():
        df_dev = df_raw[df_raw["Device number"] == dev_num].copy()
        if df_dev.empty:
            print(f" Warning: device {dev_num} has no samples.")
            continue

        # Ensure numeric for safety
        for col in EXG_COLS + IMU_COLS + ["Timestamp"]:
            if col in df_dev.columns:
                df_dev[col] = pd.to_numeric(df_dev[col], errors="coerce")

        # EXG stream
        if role_name == "EEG_1":
            # EEG_1: only Channel1–Channel6
            exg_cols = ["Timestamp"] + [f"Channel{i}" for i in range(1, 7)]
        else:
            # EMG_2 and EMG_3: all 8 channels
            exg_cols = ["Timestamp"] + EXG_COLS

        exg_df = df_dev[exg_cols].dropna(subset=["Timestamp"])
        streams[role_name] = exg_df

        # IMU stream for this device
        imu_role = IMU_ROLES[dev_num]
        imu_cols = ["Timestamp"] + IMU_COLS
        imu_df = df_dev[imu_cols].dropna(subset=["Timestamp"])
        streams[imu_role] = imu_df

    return streams
def get_structured_device_dir(subject_id: str):
    """
    Ensure and return:
      MindRove_Data/structured_data/Sxx/by_device/
    """
    out_dir = os.path.join(STRUCTURED_ROOT, subject_id, "by_device")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
def extract_clean_trial_id(filename: str) -> str:
    """
    Convert a full raw filename like:
        'S04_T01_29_11_25_19_27_24.csv'
    into a clean trial ID:
        'S04_T01'

    Assumes the base name starts with:
        <SubjectID>_<TrialID>_...
    e.g., S04_T01_...
    """
    base = os.path.splitext(filename)[0]  # remove .csv
    parts = base.split("_")
    if len(parts) >= 2:
        clean = parts[0] + "_" + parts[1]  # e.g. S04 + T01
    else:
        # Fallback: use the whole base name if unexpected format
        clean = base
    return clean
for subject_id in subjects:
    print("\n============================")
    print(f"Processing subject: {subject_id}")
    print("============================")

    raw_files, trig_files = list_trials_for_subject(subject_id)
    print(f"  Raw trials:     {len(raw_files)}")
    print(f"  Trigger trials: {len(trig_files)}")

    out_dir = get_structured_device_dir(subject_id)

    for raw_path in raw_files:
        # Original filename, e.g. 'S04_T01_29_11_25_19_27_24.csv'
        filename = os.path.basename(raw_path)

        # Clean trial ID, e.g. 'S04_T01'
        clean_trial = extract_clean_trial_id(filename)

        print(f"  → Splitting {clean_trial} (from {filename})")

        # Load raw MindRove CSV with our custom loader
        df_raw = load_mindrove_csv_simple(raw_path)

        # Split into 6 streams: EEG_1, EMG_2, EMG_3, IMU_1, IMU_2, IMU_3
        streams = split_devices(df_raw)

        # Save each stream as a clean CSV
        for stream_name, df_stream in streams.items():
            # Example: S04_T01_EEG_1_raw.csv
            out_name = f"{clean_trial}_{stream_name}_raw.csv"
            out_path = os.path.join(out_dir, out_name)
            df_stream.to_csv(out_path, index=False)

    print(f"✓ Finished subject {subject_id}")

