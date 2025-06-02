import warnings
warnings.simplefilter('ignore')

import random 
import os
import numpy as np
import torch

def seed_everything(seed_num):
    
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
