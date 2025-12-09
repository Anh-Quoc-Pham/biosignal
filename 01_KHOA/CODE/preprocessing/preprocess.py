PREPROCESSED_ROOT = "/content/drive/MyDrive/MindRove_Data/preprocessed"

FS = 500  # Hz (sampling rate for all signals)

# Choose which subject to work with
SUBJECT_ID = "S04"   # <- change if needed (e.g., "S01", "S02", ...)

# -----------------------------
# Saving options
# -----------------------------
SAVE_EEG = False   # Save filtered EEG? (we only visualize raw/filterd here)
SAVE_EMG = True    # Save filtered EMG for ML? (we are going to use EMG data for the next tutorial)
SAVE_IMU = False   # Save IMU outputs? (we only visualize raw here)

print("STRUCTURED_ROOT  :", STRUCTURED_ROOT)
print("PREPROCESSED_ROOT:", PREPROCESSED_ROOT)
print("SUBJECT_ID       :", SUBJECT_ID)