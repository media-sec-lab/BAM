import os
from tqdm import tqdm
from utils import *

import glob
import librosa
import soundfile
import numpy as np
import shutil

def preprocess(root, cache_root, data_type, samplerate):
    raw_list = glob.glob(f'{root}/{data_type}/**/*.wav',recursive=True)
    os.makedirs(f'{cache_root}/raw/{data_type}',exist_ok=True)

    for path in tqdm(raw_list, desc='Resamplping...'):
        data ,sr = librosa.load(path, sr=samplerate)
        name = os.path.basename(path).split('.')[0]
        sp = f'{cache_root}/raw/{data_type}/{name}.wav'
        soundfile.write(sp, data, samplerate)

def get_boundary_labels(root, type, cache_root, resolution=0.02):
    data_root = os.path.join(root, f'{type}/con_wav')
    labels_path = os.path.join(root, f'segment_labels/{type}_seglab_{resolution}.npy')
    labels = np.load(labels_path, allow_pickle=True).item() 
    utt_list = glob.glob(f'{data_root}/**/*.wav',recursive=True)

    shutil.copy(labels_path, os.path.join(f'{cache_root}/{type}_seglab_{resolution}.npy'))
    save_dir = f'{cache_root}/boundary_{resolution}_labels/{type}'
    os.makedirs(save_dir,exist_ok=True)
    all_count = 0
    boundary_count = 0
    for utt in tqdm(utt_list, desc=f'Get {type} boundary label...'):
        name = os.path.basename(utt).split('.')[0]
        utt_label = labels[name].astype(np.int32)
        all_count += len(utt_label)
        pos = []
        for i, label in enumerate(utt_label):
            if i == 0:
                last = label
            if label != last:
                splice_index = i if label==0 else i-1 
                pos.append(splice_index)
                last = label
                
        pos = list(set(pos))
        boundary_count += len(pos)
        boundary_label= np.zeros_like(utt_label)
        boundary_label[pos] = 1.0
        np.save(f'{save_dir}/{name}_boundary.npy',boundary_label)
    print(f'pos_weigth: {(all_count-boundary_count)/(boundary_count)}')

if __name__ == '__main__':
    partialspoof_path = "/pubdata/zhongjiafeng/PartialSpoofV1.2/database"
    data_cache_path = "./data"
    for data_type in ['train','dev','eval']:
        preprocess(partialspoof_path, data_cache_path, data_type, 16000)
        get_boundary_labels(partialspoof_path, data_type, data_cache_path, resolution=0.16)