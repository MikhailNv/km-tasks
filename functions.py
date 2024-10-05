import os
import numpy as np
import pandas as pd
import cv2


def extract_ppg_signal(frame_folder_path):
    frame_files = sorted(
        [f for f in os.listdir(frame_folder_path) if f.endswith((".jpg", ".png"))]
    )
    red_channel = []
    green_channel = []
    blue_channel = []
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        # BGR
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        blue_channel.append(b_mean)
        green_channel.append(g_mean)
        red_channel.append(r_mean)
    # Take average of the 3 channels
    ppg_signal = (
        np.array(red_channel) + np.array(green_channel) + np.array(blue_channel)
    ) / 3
    return ppg_signal


def get_windows_and_labels(ppg_signal, gt_df, window_sec=5, fps=25):
    count_gt = len(gt_df.index)
    count_windows = count_gt // window_sec
    windows = []
    labels = []
    window_size = window_sec * fps
    for i in range(count_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = ppg_signal[start_idx:end_idx]
        windows.append(window)
        # Get ground truth for the window
        gt_start = i * window_sec
        gt_end = gt_start + window_sec - 1
        gt_values = gt_df.loc[gt_start:gt_end, "SpO2"]
        label = np.mean(gt_values)  # truth value for this window
        labels.append(label)
    return windows, labels


def load_data(data_path):
    X = []  # input values in CNN
    y = []  # truth output values for CNN
    subjects = os.listdir(data_path)
    for subject in subjects:
        face_path = os.path.join(data_path, subject, "Face")
        gt_path = os.path.join(data_path, subject, "gt_SpO2.csv")
        gt_df = pd.read_csv(gt_path)  # dataframe with SpO2 values
        # Extract PPG signal
        ppg_signal = extract_ppg_signal(face_path)
        # Break down signals into windows
        windows, labels = get_windows_and_labels(ppg_signal, gt_df)
        X.extend(windows)
        y.extend(labels)
    return np.array(X), np.array(y)