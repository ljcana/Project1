# import modules

import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# setting up environment parameters

FRAME_TIME = 0.1    # time interval
GRAVITY_ACCEL = 0.12    # gravity constant
BOOST_ACCEL = 0.18  # thrust constant
