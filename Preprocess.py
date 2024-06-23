import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt
import datetime

def preprocess_frame(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    # Convert to grayscale and downsample the image
    frame = frame[35:195]  # Crop
    frame = frame[::2, ::2]  # Downsample by factor of 2
    frame = frame.mean(axis=2)  # Convert to grayscale
    frame = frame / 255.0  # Normalize
    frame = frame[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    return frame
