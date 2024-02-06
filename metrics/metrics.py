import numpy
import math

import os
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../mem_nets'))
from assessor.memnet import memnet
sys.path.append(os.path.join(os.path.dirname(__file__), '../../mem_nets'))
from amnet.amnet import AMNet, PredictionResult
from amnet.config import get_amnet_config


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))
