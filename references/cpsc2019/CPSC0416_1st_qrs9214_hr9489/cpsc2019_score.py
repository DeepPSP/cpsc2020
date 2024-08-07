import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import re
import math
from scipy import io
from CPSC2019_challenge import *
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    from keras.models import model_from_json
except:
    from tensorflow.keras.models import model_from_json

def pp(data):
    x = np.max(data)
    if x>20:  # 20 mV
        b = np.argwhere(data>20)
        for k in b[:,0]:
            if k>0:
                data[k] = data[k-1]
    return data

def load_ans(data_path_, rpos_path_, fs_):
    def is_mat(l):
        return l.endswith('.mat')
    ecg_files = list(filter(is_mat, os.listdir(data_path_)))
    rpos_files = list(filter(is_mat, os.listdir(rpos_path_)))
    model1 = model_from_json(open('./models/CNN.json').read())
    model1.load_weights('./models/CNN.h5')
    model2 = model_from_json(open('./models/CRNN.json').read())
    model2.load_weights('./models/CRNN.h5')
    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []
    for rpos_file in rpos_files:
        index = re.split('[_.]', rpos_file)[1]
        ecg_file = 'data_' + index + '.mat'

        ref_path = os.path.join(rpos_path_, rpos_file)
        ecg_path = os.path.join(data_path_, ecg_file)

        ecg_data = io.loadmat(ecg_path)['ecg'].squeeze()
        ecg_data = pp(ecg_data)
        r_ref = io.loadmat(ref_path)['R_peak'].squeeze()
        r_ref = r_ref[(r_ref >= 0.5*fs_) & (r_ref <= 9.5*fs_)]

        r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * fs_ and loc < len(ecg_data) - 0.5 * fs_)])
        result1 = model1.predict(ecg_data.reshape(1,5000,1))
        result2 = model2.predict(ecg_data.reshape(1,5000,1))
        result = (result1[0,:,0]+result2[0,:,0])/2
        hr_ans, r_ans = CPSC2019_challenge(result)
        try:
            hr_ans, r_ans = CPSC2019_challenge(result)
        except:
            hr_ans = 80
            r_ans = np.array([0])
        HR_ref.append(round( 60 * fs_ / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        HR_ans.append(hr_ans)
        R_ans.append(r_ans)

    return R_ref, HR_ref, R_ans, HR_ans

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0

        if math.isnan(hr_ans[i]):
            hr_ans[i] = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5*fs_ + thr_*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
            elif j == len(r_ref[i])-1:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= 9.5*fs_ - thr_*fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= r_ref[i][j+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if FN + FP > 1:
            record_flags[i] = 0
        elif FN == 1 and FP == 0:
            record_flags[i] = 0.3
        elif FN == 0 and FP == 1:
            record_flags[i] = 0.7

    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    print( 'QRS_acc: {}'.format(rec_acc))
    print('HR_acc: {}'.format(hr_acc))
    print('Scoring complete.')

    return rec_acc, hr_acc

if __name__ == '__main__':
    FS = 500
    THR = 0.075

    DATA_PATH = './data/cpsc2019/'
    RPOS_PATH = './data/cpsc2019/ref/'

    R_ref, HR_ref, R_ans, HR_ans = load_ans(DATA_PATH, RPOS_PATH, FS)
    rec_acc, hr_acc = score(R_ref, HR_ref, R_ans, HR_ans, FS, THR)

    with open('score.txt', 'w') as score_file:
        print('Total File Number: %d\n' %(np.shape(HR_ans)[0]), file=score_file)
        print('R Detection Acc: %0.4f' %rec_acc, file=score_file)
        print('HR Detection Acc: %0.4f' %hr_acc, file=score_file)

        score_file.close()
