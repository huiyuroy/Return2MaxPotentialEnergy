import os
import math
import random
import operator
import pickle
import numpy as np
from abc import ABC, abstractmethod

EPS = 1e-6
PI = np.pi
PI_2 = np.pi * 2
PI_1_2 = np.pi * 0.5
PI_1_4 = np.pi * 0.25
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180
