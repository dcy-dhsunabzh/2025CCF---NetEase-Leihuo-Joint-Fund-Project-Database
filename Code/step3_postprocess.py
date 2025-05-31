from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

def savitzky_golay_filter(data, window_length=7, polyorder=3):
    if window_length >= data.shape[0]:
        window_length = data.shape[0] - 1 if data.shape[0] % 2 == 0 else data.shape[0]
    if window_length % 2 == 0:
        window_length -= 1
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)


def eye_close_fix(data, threshold=0.25):
    # Left eye fix
    left_eye_distance = data[:, 18] + data[:, 14]
    rows = (left_eye_distance <= threshold)
    data[rows, 18] = -data[rows, 14]
    data[rows, 17] = 0.0
    
    # Right eye fix
    right_eye_distance = data[:, 34] + data[:, 30]
    rows = (right_eye_distance <= threshold)
    data[rows, 34] = -data[rows, 30]
    data[rows, 33] = 0.0
    
    return data


if __name__ == '__main__':
    input_dir = Path("./mid_data/rig")
    output_dir = Path("./data/result")
    output_dir.mkdir(parents=True, exist_ok=True) 
    
    for input_file in tqdm(input_dir.iterdir()):
        output_file = output_dir / input_file.name
        data = np.loadtxt(input_file, delimiter=',')

        data = eye_close_fix(data)
        data = savitzky_golay_filter(data, window_length=13, polyorder=3)
        data = eye_close_fix(data)
        
        np.savetxt(output_file, data, fmt='%.6f', delimiter=',')
