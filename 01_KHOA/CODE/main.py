from data_loading.loader import (
list_subjects,
process_all_subjects
)
from preprocessing.preprocess import (
    preprocess_eeg, 
    preprocess_emg, 
    preprocess_imu
)
from preprocessing.visualization import (
    plot_eeg_demo, 
    plot_emg_envelope, 
    plot_imu_demo
)
from config.path import SUBJECTS, DATA_ROOT

def main():
    print("Choose step you want to do:")
    print("1 - Loading data")
    print("2 - Preprocess")
    print("3 - Visualize")
    print("0 - Do alls")

    
    choice = input("Please input your selection: ")
    choices = [int(c.strip()) for c in choice.split(",")]
    subjects = list_subjects(DATA_ROOT)
    for subject_id in SUBJECTS:
        print(f"\nProcessing {subject_id}...")

        eeg_demo_raw, eeg_demo_filt, eeg_t = None, None, None
        env_aligned = None
        imu_demo = None

        # ------------------------------
        # Loading
        # ------------------------------
        if 1 in choices or 0 in choices:
            print("  -> Loading")
            process_all_subjects(subjects)
            
         # ------------------------------
        # Preprocessing
        # ------------------------------
        if 2 in choices or 0 in choices:
            print("  -> Preprocessing")
            eeg_demo_raw, eeg_demo_filt, eeg_t = preprocess_eeg(subject_id, save=False)
            env_aligned = preprocess_emg(subject_id, save=True)
            imu_demo = preprocess_imu(subject_id, save=False)

        if 3 in choices or 0 in choices:
            print("  -> Visualizing")
            plot_eeg_demo(eeg_demo_raw, eeg_demo_filt, eeg_t, subject_id)
            plot_emg_envelope(env_aligned, subject_id)
            plot_imu_demo(imu_demo)
            

        # ------------------------------
        # Model training
        # ---------------xs---------------
                

if __name__ == "__main__":
    main()