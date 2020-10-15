

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import transformers.modeling_roberta

class RobertaForCloth():
    def __init__(self):
        print('inited')