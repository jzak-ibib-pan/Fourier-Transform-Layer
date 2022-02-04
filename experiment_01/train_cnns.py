import os
# turn off gpu - to separate function
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, MaxPooling2D, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16
from tensorflow.keras.initializers import ones
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
import json
import cv2
import time
import datetime as dt
from pandas import DataFrame as df
import ipykernel

from contextlib import redirect_stdout
from ast import literal_eval
from matplotlib import pyplot as plt
from artykul_00 import FTL, FTL_01

# session reset
#from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K