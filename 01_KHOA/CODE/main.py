import os
from data_loading.loader import (
    list_subjects,
    list_trials_for_subject, 
    load_mindrove_csv_simple,
    extract_clean_trial_id,
    split_devices,
    out_dir
)

from config.path import DATA_ROOT

subjects = list_subjects(DATA_ROOT)
print("Detected subjects:", subjects)
for sid in subjects:
    raw_files, trig_files = list_trials_for_subject(sid)
    print(f"{sid}: {len(raw_files)} raw, {len(trig_files)} triggers")
    for raw_path in raw_files:

            filename = os.path.basename(raw_path)
            clean_trial = extract_clean_trial_id(filename)

            df_raw = load_mindrove_csv_simple(raw_path)
            streams = split_devices(df_raw)
            # Save all 6 streams
            for stream_name, df_stream in streams.items():
                out_name = f"{clean_trial}_{stream_name}_raw.csv"
                out_path = os.path.join(out_dir, out_name)
                df_stream.to_csv(out_path, index=False)
print("✓ ALL SUBJECTS PROCESSED SUCCESSFULLY!")