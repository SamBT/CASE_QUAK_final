import multiprocessing
import sys
import math
import time
import tqdm
import numpy as np
import scipy as sp
from scipy import stats
from scipy.spatial import ConvexHull
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from sklearn import metrics
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.utils.data as utils

from argparse import ArgumentParser
import re
import os
import json

#sys.path.append("../new_flows")
#from flows import RealNVP, Planar, MAF
#from models import NormalizingFlowModel
from nflows.flows.base import Flow
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseQuadraticAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
