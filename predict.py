from tensorflow.keras.models import load_model
import numpy as np
from Code.extract_feature import *
from Code.Model import TIMNET_Model
from Code.TIMNET import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
parser.add_argument('--data', type=str, default='EMODB')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

# Load the model
model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
model.load_weights('Models/EMOVO_46_2024-03-22_14-39-44/10-fold_weights_best_1.hdf5')

# Load and preprocess the audio file
audio = extract_feature.extract('path/to/audio.wav')

# Predict the emotion
predictions = model.predict(np.array([audio]))
predicted_emotion = CLASS_LABELS[np.argmax(predictions[0])]

print(predicted_emotion)