import os
from glob import glob, iglob
import numpy as np
import pandas as pd
import cv2


__all__ = [
    "Data2PPG"
]


class Data2PPG:
    @classmethod
    def __get_ppg_signal(cls, images_path: str):
        """ PPG Signal Acquisition Method
        Args:
            images_path: str - Images generator object
        Returns:
            ppg_signal: NDArray - PPG signal array
        """
        rc = []
        gc = []
        bc = []
        for frame_file in images_path:
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            b_mean = np.mean(frame[:, :, 0])
            g_mean = np.mean(frame[:, :, 1])
            r_mean = np.mean(frame[:, :, 2])
            rc.append(r_mean)
            gc.append(g_mean)
            bc.append(b_mean)
        ppg_signal = (
            np.array(rc) + np.array(gc) + np.array(bc)
        ) / 3
        return ppg_signal

    @classmethod
    def __get_windows_and_labels(cls, ppg_signal, csv_df, window_sec=5, fps=25):
        """ Method of receiving signals of different time ranges
        Args:
            ppg_signal: NDArray - PPG signal array
            csv_df: DataFrame - SpO2 each subject DataFrame
            window_sec: int - window size
            fps: int - Frames Per Second 
        Returns:
            (windows, labels): Tuple - Tuple of found windows and labels
        """
        count_gt = len(csv_df.index)
        count_windows = count_gt // window_sec
        windows = []
        labels = []
        window_size = window_sec * fps
        for window_idx in range(count_windows):
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size
            window = ppg_signal[start_idx:end_idx]
            windows.append(window)
            gt_start = window_idx * window_sec
            gt_end = gt_start + window_sec - 1
            gt_values = csv_df.loc[gt_start:gt_end, "SpO2"]
            label = np.mean(gt_values)
            labels.append(label)
        return windows, labels

    @classmethod
    def get_metrics(cls, data_path: str):
        """ Method of receiving X and y metrics
        Args:
            data_path: str - Path to Train/Validate/Test data
        Returns:
            (X, y): Tuple - Tuple of found metrics
        """
        X_values = []
        y_values = []
        subjects = os.listdir(data_path)
        for subject in subjects:
            images_path = glob(f"{data_path}/{subject}/**/*.jpg")
            csv_path = glob(f"{data_path}/{subject}/gt_SpO2.csv").pop()
            csv_df = pd.read_csv(csv_path)
            ppg_signal = cls.__get_ppg_signal(images_path)
            windows, labels = cls.__get_windows_and_labels(ppg_signal, csv_df)
            X_values.extend(windows)
            y_values.extend(labels)
        return np.array(X_values), np.array(y_values)