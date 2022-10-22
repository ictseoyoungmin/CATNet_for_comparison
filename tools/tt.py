# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""

import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)
for pt in sys.path:
    print(pt)
import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import gc
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset

