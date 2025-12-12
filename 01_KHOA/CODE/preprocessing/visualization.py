import matplotlib.pyplot as plt
import numpy as np
from preprocess import FS

def plot_eeg_demo(eeg_demo_raw, eeg_demo_filt, eeg_t, subject_id):
    if eeg_demo_raw is None:
        print("No EEG demo to plot.")
        return

    fig, ax = plt.subplots(2,1, figsize=(12,7), sharex=True)
    ax[0].plot(eeg_t, eeg_demo_raw)
    ax[0].set_title(f"{subject_id} – EEG_1 – Channel1 (Raw)")
    ax[0].set_ylabel("Amplitude (µV)")
    ax[0].grid(True)

    ax[1].plot(eeg_t, eeg_demo_filt)
    ax[1].set_ylabel("Amplitude (µV)")
    ax[1].set_title(f"{subject_id} – EEG_1 – Channel1 (Filtered)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_emg_envelope(env_aligned, subject_id):
    if env_aligned is None:
        print("No EMG envelope to plot.")
        return
    t_env = np.arange(env_aligned.shape[1]) / FS
    mean_env = env_aligned.mean(axis=0)
    std_env  = env_aligned.std(axis=0)

    plt.figure(figsize=(12,5))
    plt.plot(t_env, mean_env, label="Mean Envelope")
    plt.fill_between(t_env, mean_env - std_env, mean_env + std_env, alpha=0.3)
    plt.title(f"{subject_id} – EMG_2 Channel1 Envelope (Mean ± Std)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_imu_demo(imu_demo):
    if imu_demo is None:
        print("No IMU demo to plot.")
        return

    ts = imu_demo["Timestamp"].to_numpy(float)
    t = (ts - ts[0]) / FS

    plt.figure(figsize=(12,4))
    plt.plot(t, imu_demo["GyroX"], label="GyroX")
    plt.plot(t, imu_demo["GyroY"], label="GyroY")
    plt.plot(t, imu_demo["GyroZ"], label="GyroZ")
    plt.title("IMU_1 Gyroscope (Raw)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(t, imu_demo["AccX"], label="AccX")
    plt.plot(t, imu_demo["AccY"], label="AccY")
    plt.plot(t, imu_demo["AccZ"], label="AccZ")
    plt.title("IMU_1 Accelerometer (Raw)")
    plt.grid(True)
    plt.legend()
    plt.show()
