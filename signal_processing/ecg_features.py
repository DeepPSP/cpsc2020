"""
References:
-----------
[1] https://github.com/mondejar/ecg-classification
[2] to add
"""
import pywt
import numpy as np

from cfg import FeatureCfg


__all__ = ["compute_ecg_features"]


def compute_ecg_features(sig:np.ndarray, rpeaks:np.ndarray) -> np.ndarray:
    """
    """
    beats = []
    for r in rpeaks:
        beats.append(filtered_ecg[r-beat_winL:r+beat_winR])
    features = np.empty((len(beats), 0))

    if 'wavelet' in FeatureCfg.features:
        tmp = []
        for beat in beats:
            tmp.append(np.array(compute_wavelet_descriptor(beat, 'db1', 3)))
        features = np.concatenate((features, np.array(tmp)), axis=1)
    if 'rr' in FeatureCfg.features:
        tmp = compute_rr_descriptor(rpeaks)
        features = np.concatenate((features, tmp))
    if 'morph' in FeatureCfg.features:
        tmp = []
        for beat in beats:
            tmp.append(np.array(compute_morph_descriptor(beat)))
        features = np.concatenate((features, np.array(tmp)), axis=1)

    return features


def compute_wavelet_descriptor(beat:np.ndarray, family:str='db1', level:int=3) -> np.ndarray:
    """
    """
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]


def compute_rr_descriptor(rpeaks:np.ndarray) -> np.ndarray:
    """
    """
    pre_R = np.array([])
    post_R = np.array([])
    local_R = np.array([])
    global_R = np.array([])

    # Pre_R and Post_R
    pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, rpeaks[1] - rpeaks[0])

    for i in range(1, len(rpeaks)-1):
        pre_R = np.append(pre_R, rpeaks[i] - rpeaks[i-1])
        post_R = np.append(post_R, rpeaks[i+1] - rpeaks[i])

    pre_R[0] = pre_R[1]
    pre_R = np.append(pre_R, rpeaks[-1] - rpeaks[-2])  

    post_R = np.append(post_R, post_R[-1])

    # Local_R: AVG from last 10 pre_R values
    for i in range(0, len(rpeaks)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j+i >= 0:
                avg_val = avg_val + pre_R[i+j]
                num = num +1
        local_R = np.append(local_R, avg_val / float(num))

	# Global R AVG: from full past-signal
    # TODO: AVG from past 5 minutes = 108000 samples
    global_R = np.append(global_R, pre_R[0])    
    for i in range(1, len(rpeaks)):
        num = 0
        avg_val = 0

        for j in range( 0, i):
            if (rpeaks[i] - rpeaks[j]) < 108000:
                avg_val = avg_val + pre_R[j]
                num = num + 1
        #num = i
        global_R = np.append(global_R, avg_val / float(num))
    features_RR = np.column_stack((pre_R, post_R, local_R, global_R))
            
    return features_RR


def compute_morph_descriptor(beat:np.ndarray) -> np.ndarray:
    """
    """
    R_pos = int((FeatureCfg.beat_winL + FeatureCfg.beat_winR) / 2)

    R_value = beat[R_pos]
    morph = np.zeros((4,))
    y_values = np.zeros(4)
    x_values = np.zeros(4)
    # Obtain (max/min) values and index from the intervals
    [x_values[0], y_values[0]] = max(enumerate(beat[0:45]), key=operator.itemgetter(1))
    [x_values[1], y_values[1]] = min(enumerate(beat[85:95]), key=operator.itemgetter(1))
    [x_values[2], y_values[2]] = min(enumerate(beat[110:120]), key=operator.itemgetter(1))
    [x_values[3], y_values[3]] = max(enumerate(beat[170:200]), key=operator.itemgetter(1))
    
    x_values[1] = x_values[1] + 85
    x_values[2] = x_values[2] + 110
    x_values[3] = x_values[3] + 170
    
    # Norm data before compute distance
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))
    
    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)
                
    for n in range(0,4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n]) 
        y_diff = R_value - y_values[n]
        morph[n] =  np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))
    
    if np.isnan(morph[n]):
        morph[n] = 0.0

    return morph
