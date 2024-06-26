import numpy as np
import os
import sys
from typing import Tuple
from tqdm import tqdm
import librosa
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,default="Code/SER_WAV_DATA/Output/temp_audio.wav", help='Path to the directory containing audio files')
parser.add_argument('--output_dir', type=str, required=True,default="Code/MFCC", help='Path to the directory where features will be saved')
parser.add_argument('--mean_signal_length', type=int, default=96000, help='Mean signal length for padding/trimming')
args = parser.parse_args()

def get_feature(file_path: str, mean_signal_length: int = 96000, embed_len: int = 39):
    signal, fs = librosa.load(file_path)  # Default setting on sampling rate
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len)
    feature = np.transpose(mfcc)
    return feature

def extract_features(data_dir: str, output_dir: str, mean_signal_length: int = 96000, embed_len: int = 39):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} directory created.")

    features = []
    current_dir = os.getcwd()
    os.chdir(data_dir)

    for filename in tqdm(os.listdir('.')):
        if filename.endswith('.wav'):
            filepath = os.path.join(os.getcwd(), filename)
            feature_vector = get_feature(filepath, mean_signal_length, embed_len)
            features.append(feature_vector)

    os.chdir(current_dir)
    features = np.array(features)
    np.save(os.path.join(output_dir, 'PRED_features.npy'), features)
    print(f"Features saved to {os.path.join(output_dir, 'PRED_features.npy')}")

extract_features(args.data_dir, args.output_dir, args.mean_signal_length)