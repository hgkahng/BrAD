
import math
import os
import random
import shutil
import time
import warnings
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils.torchvision_wrappers as models_wrappers
from utils import domainnet
from utils.data import DataCoLoader, IndexedDataset

import moco.loader
from config import parser, setup
from moco.builder import concat_all_gather


def parse_arguments():
    pass


def main(args=None):
    pass


if __name__ == '__main__':

    main()
